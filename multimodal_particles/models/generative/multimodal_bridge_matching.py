import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from typing import List, Tuple, Dict

from multimodal_particles.models.architectures.epic import MultiModalEPiC
from multimodal_particles.models.generative.bridges import LinearUniformBridge, TelegraphBridge


# @dataclass
# class BridgeState:
#     time: torch.Tensor = None
#     continuous: torch.Tensor = None
#     discrete: torch.Tensor = None
#     absorbing: torch.Tensor = None

#     def to(self, device):
#         return BridgeState(
#             time=self.time.to(device) if isinstance(self.time, torch.Tensor) else None,
#             continuous=self.continuous.to(device)
#             if isinstance(self.continuous, torch.Tensor)
#             else None,
#             discrete=self.discrete.to(device)
#             if isinstance(self.discrete, torch.Tensor)
#             else None,
#             absorbing=self.absorbing.to(device)
#             if isinstance(self.absorbing, torch.Tensor)
#             else None,
#         )

#     @staticmethod
#     def cat(states: List["BridgeState"], dim=0) -> "BridgeState":
#         # function to concat list of states int a single state
#         def cat_attr(attr_name):
#             attrs = [getattr(s, attr_name) for s in states]
#             if all(a is None for a in attrs):
#                 return None
#             attrs = [a for a in attrs if a is not None]
#             return torch.cat(attrs, dim=dim)

#         return BridgeState(
#             time=cat_attr("time"),
#             continuous=cat_attr("continuous"),
#             discrete=cat_attr("discrete"),
#             absorbing=cat_attr("absorbing"),
#         )

@dataclass
class HybridState:
    """time-dependent hybrid bridge state (t, x, k, mask)"""

    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None

    def to(self, device: str):
        return self._apply(lambda tensor: tensor.to(device))

    def detach(self):
        return self._apply(lambda tensor: tensor.detach())

    def cpu(self):
        return self._apply(lambda tensor: tensor.cpu())

    def clone(self):
        return self._apply(lambda tensor: tensor.clone())

    @property
    def device(self):
        return (
            self.continuous.device
            if self.continuous is not None
            else self.discrete.device
        )

    @staticmethod
    def cat(states: List["HybridState"], dim=0) -> "HybridState":
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name, None) for s in states]
            if all(a is None for a in attrs):
                return None
            attrs = [a for a in attrs if a is not None]
            return torch.cat(attrs, dim=dim)

        return HybridState(
            time=cat_attr("time"),
            continuous=cat_attr("continuous"),
            discrete=cat_attr("discrete"),
            absorbing=cat_attr("absorbing"),
        )

    def _apply(self, func):
        """apply transformation function to all attributes."""
        return HybridState(
            time=func(self.time) if isinstance(self.time, torch.Tensor) else None,
            continuous=func(self.continuous)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=func(self.discrete)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            absorbing=func(self.absorbing) if isinstance(self.absorbing, torch.Tensor) else None,
        )



@dataclass
class MultiHeadOutput:
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None


class MultiModalBridgeMatching(L.LightningModule):
    """Model for hybrid data with varying size"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size.features

        self.encoder = MultiModalEPiC(config)

        self.bridge_continuous = LinearUniformBridge(config)
        self.bridge_discrete = TelegraphBridge(config)
        self.bridge_absorbing = None  # TODO implement absorbing bridge

        self.loss_continuous_fn = nn.MSELoss(reduction="none")
        self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")
        self.loss_absorbing_fn = None # TODO implement absorbing loss

        self.loss_multihead = MultiHeadLoss(mode='learnable')

        self.save_hyperparameters()

    def forward(self, state: HybridState, batch) -> MultiHeadOutput:
        continuous, discrete, absorbing = self.encoder(
            t=state.time,
            x=state.continuous,
            k=state.discrete,
            mask=state.absorbing,
            context_continuous=getattr(batch, "context_continuous", None),
            context_discrete=getattr(batch, "context_discrete", None),
        )
        return MultiHeadOutput(continuous, discrete, absorbing)

    def sample_bridges(self, batch) -> HybridState:
        """sample stochastic bridges"""
        continuous, discrete = None, None
        t = torch.rand(batch.target_continuous.shape[0], device=self.device).type_as(
            batch.target_continuous
        )

        time = self.reshape_time(t, batch.target_continuous)
        if hasattr(self, "bridge_continuous"):
            continuous = self.bridge_continuous.sample(
                time, batch.source_continuous, batch.target_continuous
            )
        if hasattr(self, "bridge_discrete"):
            discrete = self.bridge_discrete.sample(
                time, batch.source_discrete, batch.target_discrete
            )
        absorbing = batch.target_mask # TODO sample absorbing state from bridge
        return HybridState(time, continuous, discrete, absorbing)

    def loss_continuous(
        self, heads: MultiHeadOutput, state: HybridState, batch
    ) -> torch.Tensor:
        """mean square error loss for drift matching"""
        if hasattr(self, "bridge_continuous"):
            vector = heads.continuous
            targets = self.bridge_continuous.drift(
                t=state.time,
                x=state.continuous,
                x0=batch.source_continuous,
                x1=batch.target_continuous,
            ).to(self.device)
            mask = state.absorbing
            loss_mse = self.loss_continuous_fn(vector, targets) * mask
            return loss_mse.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=self.device)

    def loss_discrete(
        self, heads: MultiHeadOutput, state: HybridState, batch
    ) -> torch.Tensor:
        """cross-entropy loss for discrete state classifier"""
        if hasattr(self, "bridge_discrete"):
            logits = heads.discrete.reshape(-1, self.vocab_size)
            targets = batch.target_discrete.reshape(-1).long()
            targets = targets.to(self.device)
            mask = state.absorbing.reshape(-1)
            loss_ce = self.loss_discrete_fn(logits, targets) * mask
            return loss_ce.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=self.device)

    def simulate_dynamics(self, state: HybridState, batch) -> HybridState:
        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        time_steps = torch.linspace(
            0.0,
            1.0 - self.config.model.pipeline.time_eps,
            self.config.model.pipeline.num_timesteps,
            device=self.device,
        )
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        for time in time_steps[1:]:
            state.time = torch.full((len(batch[0]), 1), time.item(), device=self.device)
            heads = self.forward(state, batch)
            state = self.bridge_continuous.solver_step(state, heads, delta_t)
            state = self.bridge_discrete.solver_step(state, heads, delta_t)
            # TODO state = self.bridge_absorbing.solver_step(state, heads, delta_t)
        return state.detach().cpu()

    def loss_absorbing(self, heads: MultiHeadOutput, batch):
        # TODO
        pass

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (x.dim() - 1)))

    ###########################
    ### Lightning functions ###
    ###########################

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, _ = self.loss_multihead([loss_0, loss_1])
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, _ = self.loss_multihead([loss_0, loss_1])
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx) -> HybridState:
        initial_state = HybridState(
            None, batch.source_continuous, batch.source_discrete, batch.source_mask
        )
        final_state = self.simulate_dynamics(initial_state, batch)
        return final_state


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.train.optimizer.params.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.train.scheduler.params.T_max,
            eta_min=self.config.train.scheduler.params.eta_min,
            last_epoch=self.config.train.scheduler.params.last_epoch,
        )
        return [optimizer], [scheduler]


class MultiHeadLoss(nn.Module):
    """
    Combines multiple losses with `learnable` or `fixed` weights.
    """

    def __init__(self, weights=None, mode=None):
        super().__init__()
        self.mode = mode
        if mode == "learnable":
            self.weights = nn.Parameter(torch.tensor([0.0, 0.0]))
        elif mode == "fixed":
            self.weights = torch.tensor(weights if weights else [1.0, 1.0])

    def forward(self, losses) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.mode == "learnable":
            combined_loss = sum(
                torch.exp(-self.weights[i]) * losses[i] + self.weights[i]
                for i in range(len(losses))
            )
        elif self.mode == "fixed":
            combined_loss = sum(self.weights[i] * losses[i] for i in range(len(losses)))
        return combined_loss, losses

    def get_weights(self) -> List[float]:
        if self.mode == "learnable":
            return [torch.exp(-weight).item() for weight in self.weights]
        elif self.mode == "fixed":
            return self.weights.tolist()