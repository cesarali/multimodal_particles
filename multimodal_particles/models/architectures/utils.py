import torch
import math
from torch import nn
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig

class InputEmbeddings(nn.Module):
    def __init__(self, config:MultimodalBridgeMatchingConfig):
        super(InputEmbeddings, self).__init__()

        # ...dimensions:
        dim_features_continuous = config.data.dim_features_continuous
        dim_features_discrete = config.data.dim_features_discrete
        dim_context_continuous = config.data.dim_context_continuous
        dim_context_discrete = config.data.dim_context_discrete


        # ...vocab sizes for discrete data:
        vocab_size = config.data.vocab_size_features
        vocab_size_context = config.data.vocab_size_context

        # ...embedding types:
        embedding_time = config.encoder.embedding_time
        embedding_continuous = config.encoder.embedding_features_continuous
        embedding_discrete = config.encoder.embedding_features_discrete
        embedding_continuous_context = config.encoder.embedding_context_continuous
        embedding_discrete_context = config.encoder.embedding_context_discrete

        # ...embedding dimensions:
        dim_emb_time = config.encoder.dim_emb_time
        dim_emb_features_continuous = (
            config.encoder.dim_emb_features_continuous
            if config.encoder.dim_emb_features_continuous
            else dim_features_continuous
        )
        dim_emb_features_discrete = config.encoder.dim_emb_features_discrete
        dim_emb_context_continuous = (
            config.encoder.dim_emb_context_continuous
            if config.encoder.dim_emb_context_continuous
            else dim_context_continuous
        )
        dim_emb_context_discrete = config.encoder.dim_emb_context_discrete

        # ...Time embeddings:

        if embedding_time == "SinusoidalPositionalEncoding":
            self.embedding_time = SinusoidalPositionalEncoding(
                dim_emb_time, max_period=10000
            )
        elif embedding_time == "Linear":
            self.embedding_time = nn.Linear(1, dim_emb_time)
        else:
            NotImplementedError(
                "Time embedding not implemented, choose from `SinusoidalPositionalEncoding`, `KANLinear` or `Linear`"
            )

        # ...Feature embeddings:

        if dim_emb_features_continuous:
            if embedding_continuous == "Linear":
                self.embedding_continuous = nn.Linear(
                    dim_features_continuous, dim_emb_features_continuous
                )
            elif embedding_continuous is None:
                self.embedding_continuous = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous features embedding not implemented, choose from `kolmogorov-arnold`, `linear` or None"
                )

        if dim_features_discrete:
            if embedding_discrete == "Embedding":
                self.embedding_discrete = nn.Embedding(
                    vocab_size, dim_emb_features_discrete
                )
            elif embedding_discrete == "Linear":
                self.embedding_discrete = nn.Linear(
                    dim_features_discrete, dim_emb_features_discrete
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

        # ...Context embeddings:

        if dim_context_continuous:
            if embedding_continuous_context == "Embedding":
                self.embedding_continuous_context = nn.Linear(
                    dim_context_continuous, dim_emb_context_continuous
                )
            elif embedding_continuous_context is None:
                self.embedding_continuous_context = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous context embedding not implemented, use `embedding` or None"
                )

        if dim_context_discrete:
            if embedding_discrete_context == "Embedding":
                self.embedding_context_discrete = nn.Embedding(
                    vocab_size_context, dim_emb_context_discrete
                )
            elif embedding_discrete_context == "Linear":
                self.embedding_context_discrete = nn.Linear(
                    dim_context_discrete, dim_emb_context_discrete
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

    def forward(
        self, t, x, k, mask=None, context_continuous=None, context_discrete=None
    ):
        """
        Forward pass of the particle embedding.

        Arguments:
        - t: Time input of shape (batch_size, 1) or (batch_size, 1, 1)
        - x: Particle continuous features of shape (batch_size, max_num_particles, dim_continuous)
        - k: Particle discrete features of shape (batch_size, max_num_particles, dim_discrete)
        - context_continuous: Continuous context features of shape (batch_size, dim_context_continuous)
        - context_discrete: Discrete context features of shape (batch_size, dim_context_discrete)
        - mask: Binary mask of shape (batch_size, max_num_particles, 1) indicating valid particles (1) or masked particles (0)

        Returns:
        - h: Embedded particles of shape (batch_size, N, dim_hidden), masked appropriately
        - context: Embedded context of shape (batch_size, dim_context)
        """

        # ...time:

        t_emb = self.embedding_time(t.squeeze(-1))
        t_context_emb = t_emb.clone().to(t_emb.device)
        if x.ndim == 3:
            t_emb = t_emb.unsqueeze(1).repeat(
                1, x.shape[1], 1
            )  # (b, dim_time_emb) -> (b, n, dim_time_emb)

        features = [t_emb]
        context = [t_context_emb]

        # ...features:

        if hasattr(self, "embedding_continuous"):
            emb = self.embedding_continuous(x)
            features.append(emb)

        if hasattr(self, "embedding_discrete"):
            emb = self.embedding_discrete(k.squeeze(-1))
            if x.ndim == 2:
                emb = emb.squeeze(1)
            features.append(emb)

        # ...context:

        if hasattr(self, "embedding_continuous_context"):
            emb = self.embedding_continuous_context(context_continuous)
            context.append(emb)

        if hasattr(self, "embedding_discrete_context"):
            emb = self.embedding_discrete_context(context_discrete).squeeze(1)
            context.append(emb)

        features = torch.cat(
            features, dim=-1
        )  # (b, n, dim_continuous_emb + dim_discrete_emb + dim_time_emb)
        context = torch.cat(
            context, dim=-1
        )  # (b, dim_context_continuous_emb + dim_context_discrete_emb + dim_time_emb)

        return features * mask, context


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding with log-linear spaced frequencies for each dimension"""

    def __init__(self, dim, max_period=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding.squeeze()
