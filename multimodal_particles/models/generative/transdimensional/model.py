from torch import nn
from multimodal_particles.models.architectures.epic import EPiC
from multimodal_particles.config_classes.transdimensional_config_unconditional import TransdimensionalEpicConfig

class TransdimensionalEPiC(nn.Module):
    """Permutation equivariant architecture for multi-modal continuous-discrete models"""

    def __init__(self, config:TransdimensionalEpicConfig):
        super().__init__()
        self.dim_features_continuous = config.dim_features_continuous
        self.dim_features_discrete = config.dim_features_discrete
        self.vocab_size = config.vocab_size_features

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