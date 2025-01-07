import torch
from torch import nn
import lightning as L
import torch.nn.functional as F

from multimodal_particles.data.particle_clouds.jets_dataloader import (
    JetsDataloaderModule,
    JetsGraphicalStructure
)

from multimodal_particles.models.architectures.epic import EPiC
from multimodal_particles.models.generative.transdimensional.structure import Structure
from multimodal_particles.config_classes.transdimensional_config_unconditional import TransdimensionalEpicConfig
from multimodal_particles.models.generative.diffusion.noising import VP_SDE, ConstForwardRate, StepForwardRate
from multimodal_particles.models.generative.diffusion.noising import get_rate_using_x0_pred
from multimodal_particles.models.architectures.gsdm import AttnBlock, ResnetBlock, get_timestep_embedding
from multimodal_particles.models.generative.transdimensional.loss import JumpLossFinalDim 
from multimodal_particles.models.architectures.egnn_utils import (
    assert_mean_zero_with_mask,
    check_mask_correct
)

def get_forward_rate(rate_function_name, max_problem_dim, rate_cut_t):
    if rate_function_name == 'step':
        return StepForwardRate(max_problem_dim, rate_cut_t)
    elif rate_function_name == 'const':
        return ConstForwardRate(max_problem_dim, None)
    else:
        raise ValueError(rate_function_name)

def get_noise_schedule(noise_schedule_name, max_problem_dim, vp_sde_beta_min, vp_sde_beta_max): 
    if noise_schedule_name == 'vp_sde':
        # DDPM schedule is beta_min=0.1, beta_max=20
        return VP_SDE(max_problem_dim, vp_sde_beta_min, vp_sde_beta_max)
    else:
        raise ValueError(noise_schedule_name)

class TransdimensionalJumpDiffusion(L.LightningModule):
    """
    this model restructures the implementation of 

    https://arxiv.org/abs/2305.16261

    from https://github.com/andrew-cr/jump-diffusion

    """
    def __init__(
            self,
            config:TransdimensionalEpicConfig,
            datamodule:JetsDataloaderModule
        ):
        super().__init__()
        self.config = config
        self._set_up(datamodule)

    def loss(self):
        pass
        #assert (self.structure.exist == 1).all()

    def _set_up(self,datamodule:JetsDataloaderModule):
        self.structure =  Structure(datamodule.exist, datamodule.observed, datamodule)

        self.net = EpsilonPrecond(self.structure,self.config)

        self.forward_rate = get_forward_rate(
            self.config.loss_kwargs.rate_function_name,
            self.config.data.max_num_particles,
            self.config.loss_kwargs.rate_cut_t)

        self.noise_schedule = get_noise_schedule(
            self.config.loss_kwargs.noise_schedule_name,
            self.config.data.max_num_particles, 
            self.config.loss_kwargs.vp_sde_beta_min,
            self.config.loss_kwargs.vp_sde_beta_max)
        
        self.jump_diffusion_loss = JumpLossFinalDim(self.forward_rate,
                                                    self.noise_schedule,
                                                    **self.config.loss_kwargs.__dict__)

    ###########################
    ### Lightning functions ###
    ###########################

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        pass
        #self.log("train_loss", loss, on_step=True, on_epoch=True)
        #return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        pass
        #self.log("val_loss", loss, on_step=True, on_epoch=True)
        #return loss

    def configure_optimizers(self):
        pass
        #optimizer = torch.optim.Adam(
        #    self.parameters(), lr=self.config.train.lr
        #)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer,
        #    T_max=self.config.train.scheduler_params["T_max"],
        #    eta_min=self.config.train.scheduler_params["eta_min"],
        #    last_epoch=self.config.train.scheduler_params["last_epoch"],
        #)
        #return [optimizer], [scheduler]

class EpsilonPrecond(torch.nn.Module):
    def __init__(
            self, 
            structure, 
            config,
            use_fp16=-1, # not used but needed for compatibility
            **model_kwargs):
        super().__init__()
        self.structure = structure
        self.model = TransdimensionalEPiC(config, structure=structure)

    def forward(self, st_batch, ts, predict='eps', forward_rate=None,nearest_atom=None):
        xt = st_batch.get_flat_lats()  # TODO mode to relevant if statement below
        eps, *others = self.model(st_batch,ts,forward_rate,nearest_atom)
        if predict == 'eps':
            return eps, *others
        elif predict == 'x0':
            x0 = self.noise_schedule.predict_x0_from_xt(xt, eps, ts)
            return x0, *others
        else:
            raise NotImplementedError(f'predict {predict} not implemented')

class TransdimensionalEPiC(nn.Module):
    """Permutation equivariant architecture for multi-modal continuous-discrete models"""

    def __init__(self, config:TransdimensionalEpicConfig,structure):
        super().__init__()

        self.config = config
        self.structure = structure

        self.max_num_particles = config.data.max_num_particles
        self.dim_features_continuous = config.data.dim_features_continuous
        self.dim_features_discrete = config.data.dim_features_discrete
        self.vocab_size_features = config.data.vocab_size_features

        self.names_in_batch = structure.graphical_structure.names_in_batch
        self.max_num_particles = structure.graphical_structure.max_num_particles
        self.num_jets = structure.graphical_structure.num_jets
        self.name_to_index = structure.graphical_structure.name_to_index

        self.output_dim = self.dim_features_continuous + self.dim_features_discrete * self.vocab_size_features
        self.output_dim_global = config.encoder.dim_hidden_glob

        self.epic = EPiC(config)
        
        self.add_discrete_head = config.encoder.add_discrete_head
        if self.add_discrete_head:
            self.fc_layer = nn.Sequential(
                nn.Linear(
                    self.dim_features_discrete * self.vocab_size_features,
                    self.dim_features_discrete * self.vocab_size_features,
                ),
                nn.SELU(),
                nn.Linear(
                    self.dim_features_discrete * self.vocab_size_features,
                    self.dim_features_discrete * self.vocab_size_features,
                ),
            )

        self.noise_schedule = None

    def set_up(self):
        output_dim_global = self.output_dim_global
        transformer_dim = self.config.encoder.transformer_dim
        n_heads = self.config.encoder.n_heads
        n_attn_blocks = self.config.encoder.n_attn_blocks
        self.rate_use_x0_pred = self.config.encoder.rate_use_x0_pred

        if self.rate_use_x0_pred:
            self.rdim = self.structure.graphical_structure.max_num_particles
        else:
            self.rdim = 1

        self.transformer_dim = transformer_dim
        self.temb_dim = self.transformer_dim

        self.temb_net = nn.Linear(self.temb_dim, self.temb_dim)

        self.transformer_1_proj_in = nn.Linear(
            output_dim_global + 6, self.transformer_dim
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

        self.near_atom_proj = nn.Linear(self.transformer_dim, 1)

        # this is for the head that gives the vector given the nearest atom and std
        self.vec_transformer_in_proj = nn.Linear(
            output_dim_global + 6 + 1 + 2, self.transformer_dim
        )
        self.vec_attn_blocks = nn.ModuleList([
            AttnBlock(self.transformer_dim, n_heads, attn_dim_reduce=1)
            for _ in range(n_attn_blocks)
        ])

        self.vec_res_blocks = nn.ModuleList([
            ResnetBlock(channels=self.transformer_dim,
                             dropout=0, temb_channels=self.temb_dim)
            for _ in range(n_attn_blocks)
        ])
        self.vec_weighting_proj = nn.Linear(self.transformer_dim, 1)

        self.pre_auto_proj = nn.Linear(self.transformer_dim, self.transformer_dim)
        self.post_auto_proj = nn.Linear(self.transformer_dim, 2*5 + 2 + 1)
    
    def forward(
        self, st_batch, ts, nearest_atom, sample_nearest_atom=False, augment_labels=None, forward_rate=None, rnd=None
    ):
        """
        #t, x, k, mask=None, context_continuous=None, context_discrete=None,
        """
        #st_batch, ts, nearest_atom, sample_nearest_atom=False, augment_labels=None, forward_rate=None, rnd=None
        target_discrete,target_continuous,context_continuous,context_discrete,mask = self.from_st_batch_to_multimodal_bridge_databatch(st_batch)
        ts = ts.unsqueeze(-1).unsqueeze(-1)
        h,h_global = self.epic(ts, target_continuous, target_discrete, mask, context_continuous, context_discrete,output_global=True)

        return h_global
    
    def from_st_batch_to_multimodal_bridge_databatch(self,st_batch):
        self.vocab_size_features
        device = st_batch.get_device()

        target_discrete = st_batch.tuple_batch[self.name_to_index["target_discrete"]] # (B, max_num_particles,vocab_size_features) (one hot)
        max_target = torch.max(F.softmax(target_discrete),dim=-1)
        target_discrete = max_target.indices.unsqueeze(-1) # (B, max_num_particles,1)

        target_continuous = st_batch.tuple_batch[self.name_to_index["target_continuous"]]

        full_dims = st_batch.get_dims()  # (B,)
        mask = torch.arange(st_batch.gs.max_num_particles, device=device).view(1, -1) < full_dims.view(-1, 1)  # (B, n_nodes)
        mask = mask.long().unsqueeze(-1)

        if "context_continuous" in self.names_in_batch:
            context_continuous = st_batch.tuple_batch[self.name_to_index["context_continuous"]]
        else:
            context_continuous = None
        
        if "context_discrete" in self.names_in_batch:
            context_discrete = st_batch.tuple_batch[self.name_to_index["context_discrete"]]
        else:
            context_discrete = None
        
        return target_discrete,target_continuous,context_continuous,context_discrete,mask
    
class EGNNMultiHeadJump(nn.Module):
    """
        EGNN backbone that gives score then a second network on the top that gives
        the rate and nearest atom prediction and a vector 

        detach_last_layer: whether to stop grad between EGNN and head net
    """
    def __init__(
            self, 
            structure, 
            detach_last_layer, 
            rate_use_x0_pred,
            n_attn_blocks, 
            n_heads, 
            transformer_dim,
            noise_embed='ts', 
            augment_dim=-1
        ):
        super().__init__()
        self.structure = structure
        self.detach_last_layer = detach_last_layer

        args.context_node_nf = 0
        in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
        # in_node_nf is for atom types and atom charges
        # +1 for time
        dynamics_in_node_nf = in_node_nf + 1

        self.egnn_net = Jump_EGNN_QM9(
            in_node_nf=dynamics_in_node_nf, 
            context_node_nf=6,
            n_dims=3, 
            hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), 
            n_layers=args.n_layers,
            attention=args.attention, 
            tanh=args.tanh, 
            mode=args.model, 
            norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, 
            sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor, 
            aggregation_method=args.aggregation_method,
            CoM0=True, return_last_layer=True
        )

        self.rate_use_x0_pred = rate_use_x0_pred
        if self.rate_use_x0_pred:
            self.rdim = self.structure.graphical_structure.max_problem_dim
        else:
            self.rdim = 1

        self.transformer_dim = transformer_dim
        self.temb_dim = self.transformer_dim

        self.temb_net = nn.Linear(self.temb_dim, self.temb_dim)

        self.transformer_1_proj_in = nn.Linear(
            self.egnn_net.egnn.hidden_nf + 6, self.transformer_dim
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

        self.near_atom_proj = nn.Linear(self.transformer_dim, 1)

        # this is for the head that gives the vector given the nearest atom and std
        self.vec_transformer_in_proj = nn.Linear(
            self.egnn_net.egnn.hidden_nf + 6 + 1 + 2, self.transformer_dim
        )
        self.vec_attn_blocks = nn.ModuleList([
            AttnBlock(self.transformer_dim, n_heads, attn_dim_reduce=1)
            for _ in range(n_attn_blocks)
        ])

        self.vec_res_blocks = nn.ModuleList([
            ResnetBlock(channels=self.transformer_dim,
                             dropout=0, temb_channels=self.temb_dim)
            for _ in range(n_attn_blocks)
        ])
        self.vec_weighting_proj = nn.Linear(self.transformer_dim, 1)

        self.pre_auto_proj = nn.Linear(self.transformer_dim, self.transformer_dim)
        self.post_auto_proj = nn.Linear(self.transformer_dim, 2*5 + 2 + 1)

    def forward(self, st_batch, ts, nearest_atom, sample_nearest_atom=False, augment_labels=None, forward_rate=None, rnd=None):
        # if sample_nearest_atom is true then we sample the nearest atom from the predicted distribution
        # and use that for the second head network. Use this during sampling but not during training


        # ts can pass directly as (B,) and should be normalized to [0,1]
        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2) # (B, n_nodes, n_features)
        x = st_batch.tuple_batch[0]
        dims = st_batch.get_dims()
        device = st_batch.get_device()
        B, n_nodes, _ = x.shape

        assert x.shape == (B, n_nodes, 3)

        atom_mask = torch.arange(st_batch.gs.max_problem_dim).view(1, -1) < dims.view(-1, 1) # (B, n_nodes)
        atom_mask = atom_mask.to(device)

        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2) # (B, n_nodes_aug, n_nodes_aug) is 1 when both col and row are 1
        assert edge_mask.shape == (B, n_nodes, n_nodes)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=device).unsqueeze(0)
        assert diag_mask.shape == (1, n_nodes, n_nodes)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(B * n_nodes * n_nodes, 1)

        atom_mask = atom_mask.long().to(device)
        edge_mask = edge_mask.long().to(device)

        node_mask = atom_mask.unsqueeze(2)
        assert node_mask.shape == (B, n_nodes, 1)
        atom_type_one_hot = st_batch.tuple_batch[1]
        assert atom_type_one_hot.shape == (B, n_nodes, 5)
        charges = st_batch.tuple_batch[2]
        assert charges.shape == (B, n_nodes)
        charges = charges.view(B, n_nodes, 1)

        context_parts = torch.cat([
            *(st_batch.tuple_batch[i] for i in range(3, len(st_batch.tuple_batch)))
        ], dim=1)
        assert context_parts.shape == (B, 6)
        context_parts = context_parts.view(B, 1, 6).repeat(1, n_nodes, 1) # (B, n_nodes, 6)
        context_parts = context_parts * node_mask

        assert_mean_zero_with_mask(x, node_mask)
        check_mask_correct([x, atom_type_one_hot, charges, context_parts], node_mask)

        # note the time gets added on by Jump_EGNN_QM9
        xh = torch.cat([x, atom_type_one_hot, charges], dim=2)
        assert xh.shape == (B, n_nodes, 3+5+1)

        net_out, net_last_layer = self.egnn_net(
            t=ts, 
            xh=xh, 
            node_mask=node_mask, 
            edge_mask=edge_mask, 
            context=context_parts
        )
        
        assert net_out.shape == (B, n_nodes, 3+5+1)
        x_out = net_out[:, :, 0:3]
        atom_type_one_hot_out = net_out[:, :, 3:8]
        charges_out = net_out[:, :, 8:9]

        D_xt = torch.cat([
            x_out.flatten(start_dim=1),
            atom_type_one_hot_out.flatten(start_dim=1),
            charges_out.flatten(start_dim=1)
        ], dim=1)
        assert D_xt.shape == (B, n_nodes * (3+5+1))

        assert net_last_layer.shape == (B, n_nodes, self.egnn_net.egnn.hidden_nf)
        
        if self.detach_last_layer:
            net_last_layer = net_last_layer.detach()

        temb = get_timestep_embedding(ts*1000, self.temb_dim)
        temb = self.temb_net(temb) # (B, C)
        temb = temb.view(B, self.temb_dim, 1).repeat(1, 1, n_nodes) # (B, C, N)

        #==========================================================================
        # DATA APPEARS 
        h = torch.cat([
            net_last_layer,
            atom_type_one_hot,
            charges.view(B, n_nodes, 1)
        ], dim=2)
        # ==========================================================================

        assert h.shape == (B, n_nodes, self.egnn_net.egnn.hidden_nf + 6)
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
                forward_rate=forward_rate, ts=ts, max_dim=st_batch.gs.max_problem_dim
            ).view(-1, 1) # (B, 1)
        else:
            x0_dim_logits = torch.zeros((B, st_batch.gs.max_problem_dim), device=device)
            f_rate_ts = forward_rate.get_rate(None, ts).view(B, 1)

            # rate_out = rate_emb.exp() # (B, 1)
            rate_out = F.softplus(rate_emb) * f_rate_ts # (B, 1)

        near_atom_logits = self.near_atom_proj(h)[:, :, 0]
        assert near_atom_logits.shape == (B, n_nodes)

        if sample_nearest_atom:
            if rnd is None:
                nearest_atom = torch.multinomial(torch.softmax(near_atom_logits, dim=1), 1).view(-1)
            else:
                nearest_atom = rnd.multinomial(torch.softmax(near_atom_logits, dim=1), num_samples=1).view(-1)

        assert nearest_atom.shape == (B,) # index from 0 to n_nodes-1

        # create a distance matrix for the closest atom (B, n_nodes)
        distances = torch.sum( (x[torch.arange(B, device=device), nearest_atom, :].view(B, 1, 3) - x)**2, dim=-1, keepdim=True).sqrt()
        assert distances.shape == (B, n_nodes, 1)

        nearest_atom_one_hot = torch.tensor([0.0, 1.0], device=device).view(1, 1, 2).repeat(B, n_nodes, 1)
        nearest_atom_one_hot[torch.arange(B, device=device), nearest_atom, 0] = 1.0
        nearest_atom_one_hot[torch.arange(B, device=device), nearest_atom, 1] = 0.0
        assert nearest_atom_one_hot.shape == (B, n_nodes, 2)

        #==========================================================================
        # DATA APPEARS 
        vec_transformer_in = torch.cat([
            net_last_layer,
            atom_type_one_hot,
            charges.view(B, n_nodes, 1),
            distances,
            nearest_atom_one_hot
        ], dim=2)
        # ==========================================================================

        assert vec_transformer_in.shape == (B, n_nodes, self.egnn_net.egnn.hidden_nf + 6 + 1 + 2)
        vec_transformer_in = vec_transformer_in * node_mask
        vec_transformer_in = self.vec_transformer_in_proj(vec_transformer_in)
        assert vec_transformer_in.shape == (B, n_nodes, self.transformer_dim)
        vec_transformer_in = vec_transformer_in.transpose(1,2)
        assert vec_transformer_in.shape == (B, self.transformer_dim, n_nodes)
        h_vec = vec_transformer_in

        for (res_block, attn_block) in zip(self.vec_res_blocks, self.vec_attn_blocks):
            h_vec = res_block(h_vec, temb)
            h_vec = attn_block(h_vec)

        assert h_vec.shape == (B, self.transformer_dim, n_nodes)
        h_vec = h_vec.transpose(1, 2)
        assert h_vec.shape == (B, n_nodes, self.transformer_dim)

        vec_weights = self.vec_weighting_proj(h_vec) # (B, N, 1)
        assert vec_weights.shape == (B, n_nodes, 1)
        vectors = x[torch.arange(B, device=device), nearest_atom, :].view(B, 1, 3) - x
        assert vectors.shape == (B, n_nodes, 3)
        vectors = vectors * node_mask
        assert vectors.shape == (B, n_nodes, 3)
        # normalize the vectors
        vectors = vectors / (torch.sqrt(torch.sum(vectors**2, dim=-1, keepdim=True)) + 1e-3)

        auto_pos_mean_out = x[torch.arange(B, device=device), nearest_atom, :] + \
            torch.sum(vec_weights * vectors, dim=1) # (B, 3)

        pre_auto_h = self.pre_auto_proj(h_vec)
        assert pre_auto_h.shape == (B, n_nodes, self.transformer_dim)
        pre_auto_h = torch.mean(pre_auto_h, dim=1) # (B, C)
        post_auto_h = self.post_auto_proj(pre_auto_h) # (B, 2*5 + 2 + 1)

        pos_std = post_auto_h[:, 0:1].repeat(1, 3) # (B, 3)
        atom_type_mean = post_auto_h[:, 1:1+5] # (B, 5)
        atom_type_std = post_auto_h[:, 1+5:1+5+5] # (B, 5)
        charge_mean = post_auto_h[:, 1+5+5:1+5+5+1] # (B, 1)
        charge_std = post_auto_h[:, 1+5+5+1:1+5+5+1+1] # (B, 1)


        auto_mean_out = torch.cat(
            [auto_pos_mean_out, atom_type_mean, charge_mean],
        dim=1).view(B, 1, 3+5+1).repeat(1, n_nodes, 1) # (B, n_nodes, 3+5+1)
        auto_std_out = torch.cat(
            [pos_std, atom_type_std, charge_std],
        dim=1).view(B, 1, 3+5+1).repeat(1, n_nodes, 1) # (B, n_nodes, 3+5+1)

        auto_mean_out = torch.cat([
            auto_mean_out[:, :, 0:3].flatten(start_dim=1),
            auto_mean_out[:, :, 3:8].flatten(start_dim=1),
            auto_mean_out[:, :, 8:9].flatten(start_dim=1),
        ], dim=1) # (B, n_nodes * (3+5+1))

        auto_std_out = torch.cat([
            auto_std_out[:, :, 0:3].flatten(start_dim=1),
            auto_std_out[:, :, 3:8].flatten(start_dim=1),
            auto_std_out[:, :, 8:9].flatten(start_dim=1),
        ], dim=1) # (B, n_nodes * (3+5+1))

        auto_mask = st_batch.get_next_dim_added_mask(B, include_onehot_channels=True, include_obs=False) #(B, n_nodes * (3+5+1))

        auto_mean_out = auto_mask * auto_mean_out
        auto_std_out = auto_mask * auto_std_out
        
        return D_xt, rate_out, (auto_mean_out, auto_std_out), x0_dim_logits, near_atom_logits

