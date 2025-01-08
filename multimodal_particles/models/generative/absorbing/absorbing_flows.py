import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from typing import List

from multimodal_particles.models.architectures.epic import EPiCWrapper
from multimodal_particles.config_classes.absorbing_flows_config import AbsorbingConfig
from multimodal_particles.models.generative.bridges import LinearUniformBridge, TelegraphBridge, AbsorbingBridge
from multimodal_particles.models.architectures.gsdm import AttnBlock, ResnetBlock, get_timestep_embedding

@dataclass
class AbsorbingBridgeState:
    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    mask_t: torch.Tensor = None

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

class AbsorbtionRate(nn.Module):
    """ Class that handles the absortion rate from the  """

    def __init__(self, config:AbsorbingConfig):
        super().__init__()

    def set_up():
        """"""
        self.transformer_dim = transformer_dim
        self.temb_dim = self.transformer_dim

        self.temb_net = nn.Linear(self.temb_dim, self.temb_dim)

        self.transformer_1_proj_in = nn.Linear(
            output_dim_global + self.vocab_size_features, self.transformer_dim
        )

        #self.transformer_1_proj_in = nn.Linear(
        #    self.egnn_net.egnn.hidden_nf + 6, self.transformer_dim
        #)

        # these are for the head that does the rate and nearest atom prediction
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
        self.post_rate_proj = nn.Linear(self.transformer_dim, self.rdim)


    def forward(self,net_last_layer):
        """ """
        if self.detach_last_layer:
            net_last_layer = net_last_layer.detach()

        ts = ts.squeeze()
        temb = get_timestep_embedding(ts*1000, self.temb_dim)
        temb = self.temb_net(temb) # (B, C)
        temb = temb.view(B, self.temb_dim, 1).repeat(1, 1, n_nodes) # (B, C, N)

        #==========================================================================
        # DATA APPEARS 
        h = torch.cat([
            net_last_layer,
            target_discrete_one_hot,
        ], dim=2)
        # ==========================================================================
        # self.egnn_net.egnn.hidden_nf -> self.output_dim_local

        assert h.shape == (B, n_nodes, self.output_dim_local + self.vocab_size_features)
        h = self.transformer_1_proj_in(h)
        assert h.shape == (B, n_nodes, self.transformer_dim)
        h = h.transpose(1,2)
        assert h.shape == (B, self.transformer_dim, n_nodes)

        for (res_block, attn_block) in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, temb)
            h = attn_block(h)

        h = h.transpose(1, 2)
        assert h.shape == (B, n_nodes, self.transformer_dim)

        rate_emb = self.pre_rate_proj(h) # (B, N, C)
        rate_emb = torch.mean(rate_emb, dim=1) # (B, C)
        rate_emb = self.post_rate_proj(rate_emb) # (B, rdim)

        if self.rate_use_x0_pred:
            x0_dim_logits = rate_emb
            rate_out = get_rate_using_x0_pred(
                x0_dim_logits=x0_dim_logits, xt_dims=st_batch.get_dims(),
                forward_rate=forward_rate, ts=ts, max_dim=st_batch.gs.max_num_particles
            ).view(-1, 1) # (B, 1)
        else:
            x0_dim_logits = torch.zeros((B, st_batch.gs.max_num_particles), device=device)
            f_rate_ts = forward_rate.get_rate(None, ts).view(B, 1)

            # rate_out = rate_emb.exp() # (B, 1)
            rate_out = F.softplus(rate_emb) * f_rate_ts # (B, 1)

class AbsorbingEPiC(nn.Module):
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
        output_dim_global = self.encoder_output_dim_local

        transformer_dim = self.config.generator.transformer_dim
        n_heads = self.config.generator.n_heads
        n_attn_blocks = self.config.generator.n_attn_blocks
        self.rate_use_x0_pred = self.config.generator.rate_use_x0_pred
        self.detach_last_layer = self.config.generator.detach_last_layer
        self.augment_dim = self.config.generator.augment_dim

        if self.rate_use_x0_pred:
            self.rdim = self.max_num_particles
        else:
            self.rdim = 1

        self.transformer_dim = transformer_dim
        self.temb_dim = self.transformer_dim

        self.temb_net = nn.Linear(self.temb_dim, self.temb_dim)

        self.transformer_1_proj_in = nn.Linear(
            output_dim_global + self.vocab_size_features, self.transformer_dim
        )

        # these are for the head that does the rate and nearest atom prediction
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
        self.post_rate_proj = nn.Linear(self.transformer_dim, self.rdim)
        
    def _set_up_continuous_head(self):
        """ at the moment, similar to the gnn the output is directly use as the head"""
        pass

    def absorbing_head(self,state:AbsorbingBridgeState,net_out,net_last_layer):
        node_mask = mask
        #node_mask = atom_mask.unsqueeze(2)

        assert net_out.shape == (B, n_nodes, self.output_dim)
        x_out = net_out[:, :, 0:self.dim_features_continuous]
        atom_type_one_hot_out = net_out[:, :, self.dim_features_continuous:]

        D_xt = torch.cat([
            x_out.flatten(start_dim=1),
            atom_type_one_hot_out.flatten(start_dim=1),
        ], dim=1)
        assert D_xt.shape == (B, n_nodes * (self.output_dim))

        assert net_last_layer.shape == (B, n_nodes, self.output_dim_local)
        
        if self.detach_last_layer:
            net_last_layer = net_last_layer.detach()

        ts = ts.squeeze()
        temb = get_timestep_embedding(ts*1000, self.temb_dim)
        temb = self.temb_net(temb) # (B, C)
        temb = temb.view(B, self.temb_dim, 1).repeat(1, 1, n_nodes) # (B, C, N)

        #==========================================================================
        # DATA APPEARS 
        h = torch.cat([
            net_last_layer,
            target_discrete_one_hot,
        ], dim=2)
        # ==========================================================================
        # self.egnn_net.egnn.hidden_nf -> self.output_dim_local

        assert h.shape == (B, n_nodes, self.output_dim_local + self.vocab_size_features)
        h = self.transformer_1_proj_in(h)
        assert h.shape == (B, n_nodes, self.transformer_dim)
        h = h.transpose(1,2)
        assert h.shape == (B, self.transformer_dim, n_nodes)

        for (res_block, attn_block) in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, temb)
            h = attn_block(h)

        h = h.transpose(1, 2)
        assert h.shape == (B, n_nodes, self.transformer_dim)

        rate_emb = self.pre_rate_proj(h) # (B, N, C)
        rate_emb = torch.mean(rate_emb, dim=1) # (B, C)
        rate_emb = self.post_rate_proj(rate_emb) # (B, rdim)

        if self.rate_use_x0_pred:
            x0_dim_logits = rate_emb
        else:
            x0_dim_logits = torch.zeros((B, st_batch.gs.max_num_particles), device=device)

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

    def forward(self,state:AbsorbingBridgeState,batch):
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
        absorbing_head = mask  # TODO

        return OutputHeads(continuous_head, discrete_head, absorbing_head)
        
class AbsorbingFlow(L.LightningModule):
    """Model for hybrid data with varying size"""

    def __init__(self, config:AbsorbingConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size_features

        self.encoder = AbsorbingEPiC(config)

        self.bridge_continuous = LinearUniformBridge(config)
        self.bridge_discrete = TelegraphBridge(config)
        self.bridge_absorbing = AbsorbingBridge(config)  # implement absorbing bridge

        self.loss_continuous_fn = nn.MSELoss(reduction="none")
        self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")
        self.loss_absorbing_fn = nn.CrossEntropyLoss(reduction="none") # implement absorbing loss

        self.save_hyperparameters()

    def forward(self, state: AbsorbingBridgeState, batch):
        return self.encoder(state,batch)

    def sample_bridges(self, batch):
        """sample stochastic bridges"""
        t = torch.rand(
            batch.target_continuous.shape[0], device=batch.target_continuous.device
        ).type_as(batch.target_continuous)

        # ts = self.min_t + (1-self.min_t) * torch.rand((B,)) # (B,)

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
        mask = heads.absorbing

        ut = self.bridge_continuous.drift(
            t=state.time,
            x=state.continuous,
            x0=batch.source_continuous,
            x1=batch.target_continuous,
        ).to(vector.device)
        loss_mse = self.loss_continuous_fn(vector, ut) * mask
        return loss_mse.sum() / mask.sum()

    def loss_discrete(self, heads: OutputHeads, batch):
        """cross-entropy loss for discrete state classifier"""
        logits = heads.discrete
        targets = batch.target_discrete
        mask = heads.absorbing
        logits = heads.discrete.reshape(-1, self.vocab_size)
        targets = batch.target_discrete.reshape(-1).long()
        targets = targets.to(logits.device)
        mask = mask.reshape(-1)
        loss_ce = self.loss_discrete_fn(logits, targets) * mask
        return loss_ce.sum() / mask.sum()

    def loss_absorbing(self, heads: OutputHeads, batch):
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

    def training_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_continous = self.loss_continuous(heads, state, batch)
        loss_discrete = self.loss_discrete(heads, batch)
        loss = loss_continous + loss_discrete
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_continous = self.loss_continuous(heads, state, batch)
        loss_discrete = self.loss_discrete(heads, batch)
        loss = loss_continous + loss_discrete
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
            self.parameters(), lr=self.config.train.optimizer.params.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.train.scheduler.params.T_max,
            eta_min=self.config.train.scheduler.params.eta_min,
            last_epoch=self.config.train.scheduler.params.last_epoch,
        )
        return [optimizer], [scheduler]
