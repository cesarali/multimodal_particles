import torch
from torch.nn.functional import softmax,sigmoid
from torch.distributions import Categorical
from dataclasses import dataclass
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig
from multimodal_particles.config_classes.absorbing_flows_config import AbsorbingConfig
from multimodal_particles.models.generative.absorbing.states import OutputHeads,AbsorbingBridgeState

class LinearUniformBridge:
    """Conditional OT Flow-Matching for continuous states.
    This bridge is a linear interpolation between boundaries states at t=0 and t=1.
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous  target state at t=1
      - x: continuous state at time t
      - z: delta function regularizer
    """

    def __init__(self, config: MultimodalBridgeMatchingConfig):
        self.sigma = config.bridge.sigma

    def sample(self, t, x0, x1):
        x = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(x)
        std = self.sigma
        return x + std * z

    def drift(self, t, x, x0, x1):
        A = 0.0
        B = 1.0
        C = -1.0
        return A * x + B * x1 + C * x0

    def diffusion(self, t):
        return 0.0

    def solver_step(self, state:AbsorbingBridgeState, heads:OutputHeads, delta_t:float, multimodal:bool = True)->AbsorbingBridgeState:
        """Euler step for ODE solver"""
        state.continuous += delta_t * heads.continuous
        if multimodal:
            state.continuous *= heads.absorbing
        else:
            state.continuous *= state.mask_t
        return state

class SchrodingerBridge:
    """Schrodinger bridge for continuous states
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous  target state at t=1
      - x: continuous state at time t
      - z: noise
    """

    def __init__(self, config: MultimodalBridgeMatchingConfig):
        self.sigma = config.bridge.sigma

    def sample(self, t, x0, x1):
        x = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(x)
        std = self.sigma * torch.sqrt(t * (1.0 - t))
        return x + std * z

    def drift(self, t, x, x0, x1):
        A = (1 - 2 * t) / (t * (1 - t))
        B = t**2 / (t * (1 - t))
        C = -1 * (1 - t) ** 2 / (t * (1 - t))
        return A * x + B * x1 + C * x0

    def diffusion(self, t):
        return self.sigma * torch.sqrt(t * (1.0 - t))

    def solver_step(self,state:AbsorbingBridgeState, heads:OutputHeads, delta_t:float, multimodal:bool = True)->AbsorbingBridgeState:
        """Euler-Maruyama step for SDE solver"""
        diffusion = self.diffusion(delta_t)
        delta_w = torch.randn_like(state.continuous)
        state.continuous += delta_t * state.continuous + diffusion * delta_w
        if multimodal:
            state.discrete *= heads.absorbing
        else:
            state.discrete *= state.mask_t        
        return state

class TelegraphBridge:
    """Multivariate Telegraph bridge for discrete states
    - t: time
    - k0: discrete source state at t=0
    - k1: discrete  target state at t=1
    - k: discrete state at time t
    """

    def __init__(self, config: MultimodalBridgeMatchingConfig):
        self.gamma = config.bridge.gamma
        self.time_epsilon = config.bridge.time_eps
        self.vocab_size = config.data.vocab_size_features

    def sample(self, t, k0, k1):
        transition_probs = self.transition_probability(t, k0, k1)
        state = Categorical(transition_probs).sample().to(k1.device)
        if state.dim() == 2:
            state = state.unsqueeze(-1)
        return state

    def rate(self, t, k, logits):
        """t: (b, 1) time tensor
        k: (b, n, 1) current state tensor
        logits: (b, n, vocab_size) logits tensor
        """
        assert (k >= 0).all() and (
            k < self.vocab_size
        ).all(), "Values in `k` outside of bound! k_min={}, k_max={}".format(
            k.min(), k.max()
        )

        qx = softmax(
            logits, dim=2
        )  # softmax to get the transition probabilities for all states
        qy = torch.gather(
            qx, 2, k.long()
        )  # get probabilities for the current state `k`

        # ...Telegraph process rates:
        S = self.vocab_size
        t, t1 = t.squeeze(), 1.0
        wt = torch.exp(-S * self.gamma * (t1 - t))
        A = 1.0
        B = (wt * S) / (1.0 - wt)
        C = wt
        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        return rate

    def transition_probability(self, t, k0, k1):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        # ...reshape input tenors:
        t = t.squeeze()
        if k0.dim() == 1:
            k0 = k0.unsqueeze(1)  # Add an extra dimension if needed
        if k1.dim() == 1:
            k1 = k1.unsqueeze(1)

        # ...set state configurations:
        k = torch.arange(0, self.vocab_size)  # shape: (vocab_size,)
        k = k[None, None, :].repeat(k0.size(0), k0.size(1), 1).float()
        k = k.to(k0.device)

        # ...compute probabilities:
        p_k_to_k1 = self.conditional_probability(t, 1.0, k, k1)
        p_k0_to_k = self.conditional_probability(0.0, t, k0, k)
        p_k0_to_k1 = self.conditional_probability(0.0, 1.0, k0, k1)

        return (p_k_to_k1 * p_k0_to_k) / p_k0_to_k1

    def conditional_probability(self, t_in, t_out, k_in, k_out):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        S = self.vocab_size
        t_out = right_time_size(t_out, k_out).to(k_in.device)
        t_in = right_time_size(t_in, k_out).to(k_in.device)
        w_t = torch.exp(-S * self.gamma * (t_out - t_in))
        k_out, k_in = right_shape(k_out), right_shape(k_in)
        kronecker = (k_out == k_in).float()
        prob = 1.0 / S + w_t[:, None, None] * ((-1.0 / S) + kronecker)
        return prob

    def solver_step(self, state:AbsorbingBridgeState, heads:OutputHeads, delta_t:float, multimodal:bool = True)->AbsorbingBridgeState:
        """tau-leaping step for master equation solver"""
        rates = self.rate(t=state.time, k=state.discrete, logits=heads.discrete)
        assert (rates >= 0).all(), "Negative rates!"
        state.discrete = state.discrete.squeeze(-1)
        # max_rate = torch.max(rates, dim=2)[1]
        all_jumps = torch.poisson(rates * delta_t).to(state.time.device)
        jump_mask = torch.sum(all_jumps, dim=-1).type_as(state.discrete) <= 1
        diff = (
            torch.arange(self.vocab_size, device=state.time.device).view(
                1, 1, self.vocab_size
            )
            - state.discrete[:, :, None]
        )
        net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(state.discrete)
        state.discrete += net_jumps * jump_mask
        state.discrete = torch.clamp(state.discrete, min=0, max=self.vocab_size - 1)
        state.discrete = state.discrete.unsqueeze(-1)
        if multimodal:
            state.discrete *= heads.absorbing
        else:
            state.discrete *= state.mask_t
        return state

class AbsorbingBridge:
    """
    Samples the survival time of a state in case the end state is 
    absorving

    - t: time
    - k0: discrete source state at t=0
    - k1: discrete  target state at t=1
    - k: discrete state at time t
    """
    def __init__(self, config: MultimodalBridgeMatchingConfig|AbsorbingConfig):
        self.gamma_absorb = torch.tensor(config.bridge.gamma_absorb, dtype=torch.float32)  # Convert gamma to a tensor
        self.time_epsilon = config.bridge.time_eps
        self.vocab_size = 2

    def survival_probability(self,t):
        """
         \Pr\left(\mbox{killing after time}\; t\right) = \\
        \exp\left(- \int_0^t F_s ds\right) = e^{-\gamma t} \; \frac{1 - e^{\gamma(t-1)}}{ 1- e^{-\gamma}}

        Args:
            t (torch.Tensor): Input tensor of shape (B,).
        Returns:
            torch.Tensor: Probability values for each t, shape (B,).
        """
        exp_neg_gamma_t = torch.exp(-self.gamma_absorb * t)
        numerator = 1 - torch.exp(self.gamma_absorb * (t - 1))
        denominator = 1 - torch.exp(-self.gamma_absorb)
        return exp_neg_gamma_t * numerator / denominator

    def sample(self,time,target_mask):
        """
            time: (b,1,1)
            mask_1: (b, n, 1) current state tensor

        """
        B = target_mask.size(0)
        N = target_mask.size(1)
        time_ = time.repeat((1,N,1)) # (B,N,1)
        random_ = torch.rand_like(time_)
        survival_probability = self.survival_probability(time_)
        mask_t = (random_ < survival_probability ).long()

        #maskes sure that survival is only for targets who are absorved at 1
        where_alive = torch.where(target_mask) 
        mask_t[where_alive] = 1
        return mask_t

    def rate(self, t, k, logits):
        """t: (b, 1) time tensor
        
        k: (b, n, 1) current state tensor
        logits: (b, n, 1) logits tensor
        """
        SP = self.survival_probability(t).unsqueeze(1)
        return SP*sigmoid(logits)

    def solver_step(self, state: AbsorbingBridgeState, heads: OutputHeads, delta_t: float) -> AbsorbingBridgeState:
        """Simplified tau-leaping step for master equation solver."""
        # Compute rates
        rates = self.rate(t=state.time, k=state.mask_t, logits=heads.absorbing)  # [B, num_particles, 1]
        rates = rates.squeeze(-1)  # Remove singleton dimension: [B, num_particles]

        # Initialize transition probability
        transition_prob = delta_t * rates  # [B, num_particles]

        # Clamp probabilities to a maximum of 1
        transition_prob = torch.clamp(transition_prob, max=1.0)

        # Sample Bernoulli for mask_t == 0
        mask_t = state.mask_t.squeeze(-1)  # [B, num_particles]
        bernoulli_sample = torch.bernoulli(transition_prob)  # [B, num_particles]

        # Update the state
        new_mask_t = torch.where(
            mask_t == 1,  # If mask_t is already 1
            torch.ones_like(mask_t),  # Stay in state 1
            bernoulli_sample  # Transition based on Bernoulli sample
        )

        # Reshape to match original dimensions
        new_mask_t = new_mask_t.unsqueeze(-1)  # [B, num_particles, 1]
        state.mask_t=new_mask_t.long()
        return state

right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = (
    lambda t, x: t
    if isinstance(t, torch.Tensor)
    else torch.full((x.size(0),), t).to(x.device)
)
