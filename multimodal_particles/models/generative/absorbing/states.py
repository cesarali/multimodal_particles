from typing import List
import torch


from dataclasses import dataclass


@dataclass
class OutputHeads:
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None


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

    def detach(self):
        """Detach all tensors in the AbsorbingBridgeState to prevent further gradient tracking."""
        self.time = self.time.detach() if self.time is not None else None
        self.continuous = self.continuous.detach() if self.continuous is not None else None
        self.discrete = self.discrete.detach() if self.discrete is not None else None
        self.mask_t = self.mask_t.detach() if self.mask_t is not None else None
        return self

    def cpu(self):
        """Moves all tensors in the AbsorbingBridgeState to the CPU."""
        return AbsorbingBridgeState(
            time=self.time.cpu() if self.time is not None else None,
            continuous=self.continuous.cpu() if self.continuous is not None else None,
            discrete=self.discrete.cpu() if self.discrete is not None else None,
            mask_t=self.mask_t.cpu() if self.mask_t is not None else None,
        )