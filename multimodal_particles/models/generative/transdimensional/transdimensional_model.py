import torch
from torch import nn
import lightning as L

from multimodal_particles.data.particle_clouds.dataloader import (
    MultimodalBridgeDataloaderModule,
    JetsGraphicalStructure
)

from multimodal_particles.models.architectures.epic import EPiC
from multimodal_particles.models.generative.transdimensional.structure import Structure
from multimodal_particles.config_classes.transdimensional_config_unconditional import TransdimensionalEpicConfig
from multimodal_particles.models.generative.diffusion.noising import VP_SDE, ConstForwardRate, StepForwardRate
from multimodal_particles.models.generative.transdimensional.loss import JumpLossFinalDim 

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

class TransdimensionalEPiC(nn.Module):
    """Permutation equivariant architecture for multi-modal continuous-discrete models"""

    def __init__(self, config:TransdimensionalEpicConfig):
        super().__init__()
        self.dim_features_continuous = config.data.dim_features_continuous
        self.dim_features_discrete = config.data.dim_features_discrete
        self.vocab_size = config.data.vocab_size_features

        self.epic = EPiC(config)
        self.add_discrete_head = config.encoder.add_discrete_head
        if self.add_discrete_head:
            self.fc_layer = nn.Sequential(
                nn.Linear(
                    self.dim_features_discrete * self.vocab_size,
                    self.dim_features_discrete * self.vocab_size,
                ),
                nn.SELU(),
                nn.Linear(
                    self.dim_features_discrete * self.vocab_size,
                    self.dim_features_discrete * self.vocab_size,
                ),
            )

    def forward(
        self, t, x, k, mask=None, context_continuous=None, context_discrete=None
    ):
        h = self.epic(t, x, k, mask, context_continuous, context_discrete)
        continuous_head = h[..., : self.dim_features_continuous]
        discrete_head = h[..., self.dim_features_continuous :]
        absorbing_head = mask  # TODO

        if self.add_discrete_head:
            return continuous_head, self.fc_layer(discrete_head), absorbing_head
        else:
            return continuous_head, discrete_head, absorbing_head
        
class TransdimensionalJumpDiffusion(L.LightningModule):
    """
    this model restructures the implementation of 

    https://arxiv.org/abs/2305.16261

    from https://github.com/andrew-cr/jump-diffusion

    """
    def __init__(
            self,
            config:TransdimensionalEpicConfig,
            datamodule:MultimodalBridgeDataloaderModule
        ):
        super().__init__()
        self.config = config
        self._set_up()

    def loss(self):
        pass
        #assert (self.structure.exist == 1).all()

    def _set_up(self):
        #self.structure = Structure(**self.config.structure_kwargs)

        self.encoder = TransdimensionalEPiC(self.config)

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
