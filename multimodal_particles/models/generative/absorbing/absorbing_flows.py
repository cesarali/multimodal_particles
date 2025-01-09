import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from typing import List
from torch.nn.functional import one_hot
from multimodal_particles.models.architectures.epic import EPiCWrapper
from multimodal_particles.config_classes.absorbing_flows_config import AbsorbingConfig
from multimodal_particles.models.generative.bridges import LinearUniformBridge, TelegraphBridge, AbsorbingBridge
from multimodal_particles.models.architectures.gsdm import AttnBlock, ResnetBlock, get_timestep_embedding

from multimodal_particles.utils.losses import MultiHeadLoss

@dataclass
class AbsorbingBridgeState:
    """
    this is the data that one is requiered to 
    evolve a process that generates
    """
    time: torch.Tensor = None #[B,1,1]
    continuous: torch.Tensor = None #[B,num_particles,continuos_feature_dim]
    discrete: torch.Tensor = None #[B,num_particles,1]
    mask_t: torch.Tensor = None #[B,num_particles,1]

    def to(self, device):
        return AbsorbingBridgeState(
            time=self.time.to(device) if isinstance(self.time, torch.Tensor) else None,
            continuous=self.continuous.to(device)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=self.discrete.to(device)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            mask_t=self.mask_t.to(device)
            if isinstance(self.mask_t, torch.Tensor)
            else None,
        )

    @staticmethod
    def cat(states: List["AbsorbingBridgeState"], dim=0) -> "AbsorbingBridgeState":
        # function to concat list of states int a single state
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name) for s in states]
            if all(a is None for a in attrs):
                return None
            attrs = [a for a in attrs if a is not None]
            return torch.cat(attrs, dim=dim)

        return AbsorbingBridgeState(
            time=cat_attr("time"),
            continuous=cat_attr("continuous"),
            discrete=cat_attr("discrete"),
            mask_t=cat_attr("absorbing"),
        )

@dataclass
class OutputHeads:
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None

class AbsorbingGenerator(nn.Module):
    """Permutation equivariant architecture for multi-modal continuous-discrete models"""

    def __init__(self, config: AbsorbingConfig):
        super().__init__()

        self.config = config

        self.max_num_particles = config.data.max_num_particles
        self.dim_features_continuous = config.data.dim_features_continuous
        self.dim_features_discrete = config.data.dim_features_discrete
        self.vocab_size_features = config.data.vocab_size_features

        # for the moment the encoder output dim has the same dimensionalities as the final continuous and discrete logits
        self.encoder_output_dim = self.dim_features_continuous + self.dim_features_discrete * self.vocab_size_features
        self.encoder_output_dim_local = config.encoder.dim_hidden_local

        self._set_up()

    def _set_up(self):
        # particle cloud encoder
        self.epic = EPiCWrapper(self.config)

        self._set_up_continuous_head()
        self._set_up_discrete_head()
        self._set_up_absorbing_head()

    def _set_up_discrete_head(self):
        self.add_discrete_head = self.config.encoder.add_discrete_head
        if self.add_discrete_head:
            self.discrete_head_mlp = nn.Sequential(
                nn.Linear(
                    self.dim_features_discrete * self.vocab_size_features,
                    self.config.generator.discrete_head_hidden_dim,
                ),
                nn.SELU(),
                nn.Linear(
                    self.config.generator.discrete_head_hidden_dim,
                    self.dim_features_discrete * self.vocab_size_features,
                ),
            )

    def _set_up_absorbing_head(self):
        encoder_output_dim_local = self.encoder_output_dim_local
        transformer_dim = self.config.generator.transformer_dim
        n_heads = self.config.generator.n_heads
        n_attn_blocks = self.config.generator.n_attn_blocks
        self.rate_use_x0_pred = self.config.generator.rate_use_x0_pred
        self.detach_last_layer = self.config.generator.detach_last_layer
        self.augment_dim = self.config.generator.augment_dim

        self.transformer_dim = transformer_dim
        self.temb_dim = self.transformer_dim
        self.temb_net = nn.Linear(self.temb_dim, self.temb_dim)

        # the first transformer projection concatenates the 
        # encoder_output_dim_local with the one hot vector binary representation
        self.transformer_1_proj_in = nn.Linear(
            encoder_output_dim_local + 2, self.transformer_dim
        )

        # these are for the head that does the rate prediction
        self.attn_blocks = nn.ModuleList([
            AttnBlock(self.transformer_dim, n_heads, attn_dim_reduce=1)
            for _ in range(n_attn_blocks)
        ])

        self.res_blocks = nn.ModuleList([
            ResnetBlock(channels=self.transformer_dim,
                             dropout=0, temb_channels=self.temb_dim)
            for _ in range(n_attn_blocks)
        ])

        self.pre_rate_proj = nn.Linear(self.transformer_dim, self.transformer_dim)
        self.post_rate_proj = nn.Linear(self.transformer_dim, 1)
        
    def _set_up_continuous_head(self):
        """ at the moment, similar to the gnn the output is directly use as the head"""
        pass

    def absorbing_head(self,state:AbsorbingBridgeState,net_out,net_last_layer):
        mask_t = state.mask_t
        B = state.mask_t.size(0)
        n_particles = state.mask_t.size(1)

        target_discrete_one_hot = one_hot(mask_t.squeeze())

        assert net_out.shape == (B, n_particles, self.encoder_output_dim)
        assert net_last_layer.shape == (B, n_particles, self.encoder_output_dim_local)
        
        if self.detach_last_layer:
            net_last_layer = net_last_layer.detach()

        # obtain time embedings
        ts = state.time.squeeze()
        temb = get_timestep_embedding(ts*1000, self.temb_dim)
        temb = self.temb_net(temb) # (B, C)
        temb = temb.view(B, self.temb_dim, 1).repeat(1, 1, n_particles) # (B, C, N)

        h = torch.cat([net_last_layer,target_discrete_one_hot], dim=2)
        assert h.shape == (B, n_particles, self.encoder_output_dim_local + 2)
        h = self.transformer_1_proj_in(h)
        assert h.shape == (B, n_particles, self.transformer_dim)
        h = h.transpose(1,2)
        assert h.shape == (B, self.transformer_dim, n_particles)

        for (res_block, attn_block) in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, temb)
            h = attn_block(h)

        h = h.transpose(1, 2)
        assert h.shape == (B, n_particles, self.transformer_dim)

        rate_emb = self.pre_rate_proj(h) # (B, N, C)
        rate_emb = self.post_rate_proj(rate_emb) # (B, N, 2)
        x0_dim_logits = rate_emb

        return x0_dim_logits
    
    def discrete_head(self,state:AbsorbingBridgeState,net_out,net_last_layer):
        discrete_head = net_out[..., self.dim_features_continuous :]
        if self.add_discrete_head:
            return self.discrete_head_mlp(discrete_head)
        else:
            return discrete_head
    
    def continuous_head(self,state:AbsorbingBridgeState,net_out,net_last_layer):
        net_out[..., : self.dim_features_continuous]
        return net_out[..., : self.dim_features_continuous]

    def forward(self,state:AbsorbingBridgeState,batch)->OutputHeads:
        # select values
        t=state.time
        continuous=state.continuous
        discrete=state.discrete
        mask=state.mask_t
        context_continuous=getattr(batch, "context_continuous", None),
        context_discrete=getattr(batch, "context_discrete", None)
    
        net_out, net_last_layer = self.epic(t, continuous, discrete, mask, context_continuous, context_discrete,output_hidden_local=True) #(B,max_num_particles,output_dim)

        continuous_head = self.continuous_head(state,net_out,net_last_layer)
        discrete_head = self.discrete_head(state,net_out,net_last_layer)
        absorbing_head = self.absorbing_head(state,net_out,net_last_layer)

        return OutputHeads(continuous_head, discrete_head, absorbing_head)
        
class AbsorbingFlow(L.LightningModule):
    """Model for hybrid data with varying size"""

    def __init__(self, config:AbsorbingConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size_features

        self.generator = AbsorbingGenerator(config)

        self.bridge_continuous = LinearUniformBridge(config)
        self.bridge_discrete = TelegraphBridge(config)
        self.bridge_absorbing = AbsorbingBridge(config)  # implement absorbing bridge

        self.loss_continuous_fn = nn.MSELoss(reduction="none")
        self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")
        self.loss_absorbing_fn = nn.BCEWithLogitsLoss(reduction="none") # implement absorbing loss
        self.loss_multihead = MultiHeadLoss(mode='learnable',number_of_losses=3)

        self.min_t = config.bridge.time_eps

        self.save_hyperparameters()

    def forward(self, state: AbsorbingBridgeState, batch)->OutputHeads:
        return self.generator(state,batch)

    def sample_bridges(self, batch)->AbsorbingBridgeState:
        """sample stochastic bridges"""
        t = self.min_t + (1 - self.min_t) * torch.rand(
            batch.target_continuous.shape[0], device=batch.target_continuous.device
        ).type_as(batch.target_continuous)

        time = self.reshape_time(t, batch.target_continuous)

        continuous = self.bridge_continuous.sample(
            time, batch.source_continuous, batch.target_continuous
        )

        discrete = self.bridge_discrete.sample(
            time, batch.source_discrete, batch.target_discrete
        )

        mask_t = self.bridge_absorbing.sample(
            time, batch.target_mask,  # replace with absorbing bridge when implemented
        )

        return AbsorbingBridgeState(time, continuous, discrete, mask_t)

    def loss_continuous(self, heads: OutputHeads, state: AbsorbingBridgeState, batch):
        """mean square error loss for velocity field"""
        vector = heads.continuous

        ut = self.bridge_continuous.drift(
            t=state.time,
            x=state.continuous,
            x0=batch.source_continuous,
            x1=batch.target_continuous,
        ).to(vector.device)

        loss_mse = self.loss_continuous_fn(vector, ut)

        return loss_mse.sum(axis=1).mean()

    def loss_discrete(self, heads: OutputHeads, batch):
        """cross-entropy loss for discrete state classifier"""
        logits = heads.discrete
        mask = heads.absorbing

        B,num_particles,_ = logits.shape
        logits = heads.discrete.reshape(-1, self.vocab_size)
        targets = batch.target_discrete.reshape(-1).long()
        targets = targets.to(logits.device)
        loss_ce = self.loss_discrete_fn(logits, targets)
        loss_ce = loss_ce.reshape(B,num_particles)

        return loss_ce.sum(axis=1).mean()

    def loss_absorbing(self, heads: OutputHeads, batch):
        target_mask = batch.target_mask.squeeze().float()
        absorbing_loss = self.loss_absorbing_fn(heads.absorbing.reshape(-1,1),target_mask.reshape(-1,1))
        absorbing_loss = absorbing_loss.sum(axis=-1)
        absorbing_loss = absorbing_loss.mean()
        return absorbing_loss

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (x.dim() - 1)))

    ###########################
    ### Lightning functions ###
    ###########################

    def training_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_continuous = self.loss_continuous(heads, state, batch)
        loss_discrete = self.loss_discrete(heads, batch)
        loss_absorbing = self.loss_absorbing(heads, batch)
        loss, _ = self.loss_multihead([loss_continuous, loss_discrete,loss_absorbing])
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_continous = self.loss_continuous(heads, state, batch)
        loss_discrete = self.loss_discrete(heads, batch)
        loss_absorbing = self.loss_absorbing(heads, batch)
        loss = loss_continous + loss_discrete + loss_absorbing
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """generates target data from source data using trained dynamics"""
        time_steps = torch.linspace(
            0.0,
            1.0 - self.config.pipeline.time_eps,
            self.config.pipeline.num_timesteps,
            device=self.device,
        )
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        delta_t = delta_t.to(self.device)
        state = AbsorbingBridgeState(
            None,
            batch.source_continuous,
            batch.source_discrete,
            batch.source_mask,
        )
        state = state.to(self.device)
        for time in time_steps[1:]:
            state.time = torch.full((len(batch[0]), 1), time.item(), device=self.device)
            heads = self.forward(state, batch)
            state = self.bridge_continuous.solver_step(state, heads, delta_t)
            state = self.bridge_discrete.solver_step(state, heads, delta_t)
        return state

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.config.train.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **self.config.train.scheduler_params
        )
        return [optimizer], [scheduler]
