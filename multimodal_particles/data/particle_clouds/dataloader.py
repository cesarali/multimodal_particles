import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from collections import namedtuple
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig
from multimodal_particles.data.transdimensional_base import GraphicalStructureBase
from multimodal_particles.models.architectures.egnn_utils import DistributionNodes

class MultimodalBridgeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.attributes = []

        # ...source

        if hasattr(self.data.source, "continuous"):
            self.attributes.append("source_continuous")
            self.source_continuous = self.data.source.continuous

        if hasattr(self.data.source, "discrete"):
            self.attributes.append("source_discrete")
            self.source_discrete = self.data.source.discrete

        if hasattr(self.data.source, "mask"):
            self.attributes.append("source_mask")
            self.source_mask = self.data.source.mask

        # ...target

        if hasattr(self.data.target, "continuous"):
            self.attributes.append("target_continuous")
            self.target_continuous = self.data.target.continuous

        if hasattr(self.data.target, "discrete"):
            self.attributes.append("target_discrete")
            self.target_discrete = self.data.target.discrete

        if hasattr(self.data.target, "mask"):
            self.attributes.append("target_mask")
            self.target_mask = self.data.target.mask

        # ...context

        if hasattr(self.data, "context_continuous"):
            self.attributes.append("context_continuous")
            self.context_continuous = self.data.context_continuous

        if hasattr(self.data, "context_discrete"):
            self.attributes.append("context_discrete")
            self.context_discrete = self.data.context_discrete

        self.databatch = namedtuple("databatch", self.attributes)

    def __getitem__(self, idx):
        return self.databatch(*[getattr(self, attr)[idx] for attr in self.attributes])

    def __len__(self):
        return len(self.data.target)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

class MultimodalBridgeDataloaderModule:
    
    def __init__(
        self, config, dataclass, batch_size: int = None, data_split_frac: tuple = None
    ):
        self.dataclass = dataclass
        self.config = config
        self.dataset = MultimodalBridgeDataset(dataclass)

        self.data_split = (
            self.config.model.train.data_split_frac
            if data_split_frac is None
            else data_split_frac
        )
        self.batch_size = (
            self.config.model.train.batch_size if batch_size is None else batch_size
        )
        self.dataloader()

    def train_val_test_split(self, shuffle=False):
        assert (
            np.abs(1.0 - sum(self.data_split)) < 1e-3
        ), "Split fractions do not sum to 1!"
        total_size = len(self.dataset)
        train_size = int(total_size * self.data_split[0])
        valid_size = int(total_size * self.data_split[1])

        # ...define splitting indices

        idx = torch.randperm(total_size) if shuffle else torch.arange(total_size)
        idx_train = idx[:train_size].tolist()
        idx_valid = idx[train_size : train_size + valid_size].tolist()
        idx_test = idx[train_size + valid_size :].tolist()

        # ...Create Subset for each split

        train_set = Subset(self.dataset, idx_train)
        valid_set = Subset(self.dataset, idx_valid) if valid_size > 0 else None
        test_set = Subset(self.dataset, idx_test) if self.data_split[2] > 0 else None

        return train_set, valid_set, test_set

    def dataloader(self):
        print("INFO: building dataloaders...")
        print(
            "INFO: train/val/test split ratios: {}/{}/{}".format(
                self.data_split[0], self.data_split[1], self.data_split[2]
            )
        )

        train, valid, test = self.train_val_test_split(shuffle=False)
        self.train = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)
        self.valid = (
            DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=False)
            if valid is not None
            else None
        )
        self.test = (
            DataLoader(dataset=test, batch_size=self.batch_size, shuffle=False)
            if test is not None
            else None
        )

        print(
            "INFO: train size: {}, validation size: {}, testing sizes: {}".format(
                len(self.train.dataset),
                len(self.valid.dataset if valid is not None else []),
                len(self.test.dataset if test is not None else []),
            )
        )

    @staticmethod
    def random_databatch(config):
        """
        For testing this generates a random databatch with the expected 
        properties of the config, without the need of loading data and 
        should adapt from config only.
        """

        # Define the namedtuple
        ParticleData = namedtuple('ParticleData', [
            'source_continuous',
            'source_discrete',
            'source_mask',
            'target_continuous',
            'target_discrete',
            'target_mask'
        ])
        batch_size = config.model.train.batch_size
        max_num_particles = config.data.target.params.max_num_particles
        dim_continuous = config.data.dim.features_continuous
        dim_discrete = config.data.dim.features_discrete
        vocab_size = config.data.vocab_size.features

        # Create the namedtuple object with random torch.Tensors
        particle_data = ParticleData(
            source_continuous=torch.rand(batch_size, max_num_particles, dim_continuous),
            source_discrete=torch.randint(0, vocab_size, (batch_size, max_num_particles, dim_discrete)),
            source_mask=torch.randint(0, 2, (batch_size, max_num_particles, 1)),
            target_continuous=torch.rand(batch_size, max_num_particles, dim_continuous),
            target_discrete=torch.rand(batch_size, max_num_particles, dim_discrete),
            target_mask=torch.randint(0, 2, (batch_size, max_num_particles, 1))
        )
        return particle_data
    
    @staticmethod
    def update_model_config(full_config,model_config:MultimodalBridgeMatchingConfig):        
        model_config.dim_features_continuous = full_config.data.dim.features_continuous
        model_config.dim_features_discrete = full_config.data.dim.features_discrete
        model_config.dim_context_continuous = full_config.data.dim.context_continuous
        model_config.dim_context_discrete = full_config.data.dim.context_discrete

        model_config.vocab_size_features = full_config.data.vocab_size.features
        model_config.vocab_size_context = full_config.data.vocab_size.context
        return model_config

class JetsGraphicalStructure(GraphicalStructureBase):
    
    def __init__(self, max_dim, histogram):
        self.max_problem_dim = max_dim
        self.nodes_dist = DistributionNodes(histogram)

    def shapes_without_onehot(self):
        k = self.max_problem_dim
        return [torch.Size([k, 3]), torch.Size([k]), torch.Size([k]), \
                torch.Size([1]), torch.Size([1]), torch.Size([1]), \
                torch.Size([1]), torch.Size([1]), torch.Size([1])
        ]

    def shapes_with_onehot(self):
        k = self.max_problem_dim
        return [torch.Size([k, 3]), torch.Size([k, 5]), torch.Size([k]),
                torch.Size([1]), torch.Size([1]), torch.Size([1]), \
                torch.Size([1]), torch.Size([1]), torch.Size([1])
        ]

    def remove_problem_dims(self, data, new_dims):
        pos, atom_type, charge, alpha, homo, lumo, gap, mu, Cv = data


        B = pos.shape[0]
        assert pos.shape == (B, *self.shapes_with_onehot()[0])
        assert atom_type.shape == (B, *self.shapes_with_onehot()[1])
        assert charge.shape == (B, *self.shapes_with_onehot()[2])

        # for b_idx in range(B):
        #     pos[b_idx, new_dims[b_idx]:, :] = 0.0
        #     cats[b_idx, new_dims[b_idx]:, :] = 0.0
        #     ints[b_idx, new_dims[b_idx]:] = 0.0

        # pos, cats, ints = data
        device = pos.device
        new_dims_dev = new_dims.to(device)

        pos_mask = torch.arange(pos.shape[1], device=device).view(1, -1, 1).repeat(pos.shape[0], 1, pos.shape[2])
        pos_mask = (pos_mask < new_dims_dev.view(-1, 1, 1))
        pos = pos * pos_mask

        atom_type_mask = torch.arange(atom_type.shape[1], device=device).view(1, -1, 1).repeat(atom_type.shape[0], 1, atom_type.shape[2])
        atom_type_mask = (atom_type_mask < new_dims_dev.view(-1, 1, 1))
        atom_type = atom_type * atom_type_mask

        charge_mask = torch.arange(charge.shape[1], device=device).view(1, -1).repeat(charge.shape[0], 1)
        charge_mask = (charge_mask < new_dims_dev.view(-1, 1))
        charge = charge * charge_mask

        return pos, atom_type, charge, alpha, homo, lumo, gap, mu, Cv

    def adjust_st_batch(self, st_batch):
        device = st_batch.get_device()
        n_nodes = st_batch.gs.max_problem_dim
        B = st_batch.B
        dims = st_batch.get_dims()

        nan_batches = torch.isnan(st_batch.get_flat_lats()).any(dim=1).long().view(B,1)
        if nan_batches.sum() > 0:
            print('nan batches: ', nan_batches.sum())
        st_batch.set_flat_lats(torch.nan_to_num(st_batch.get_flat_lats()))


        x0_pos = st_batch.tuple_batch[0]
        assert x0_pos.shape == (B, n_nodes, 3)


        atom_mask = torch.arange(n_nodes).view(1, -1) < dims.view(-1, 1) # (B, n_nodes)
        assert atom_mask.shape == (B, n_nodes)
        atom_mask = atom_mask.long().to(device)
        node_mask = atom_mask.unsqueeze(2)
        assert node_mask.shape == (B, n_nodes, 1)

        # if any dims are 0 then set the node mask to all 1's. otherwise you get nans
        # all these results will be binned later anyway
        node_mask[dims==0, ...] = torch.ones((B, n_nodes, 1), device=device)[dims==0, ...].long()

        masked_max_abs_value = (x0_pos * (1 - node_mask)).abs().sum().item()
        assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
        N = node_mask.sum(1, keepdims=True)

        mean = torch.sum(x0_pos, dim=1, keepdim=True) / N
        assert mean.shape == (B, 1, 3)
        x0_pos = x0_pos - mean * node_mask

        assert x0_pos.shape == (B, n_nodes, 3)
        st_batch.set_flat_lats(torch.cat([
            x0_pos.flatten(start_dim=1),
            st_batch.tuple_batch[1].flatten(start_dim=1),
            st_batch.tuple_batch[2].flatten(start_dim=1)
        ], dim=1))
        return mean

    def get_auto_target(self, st_batch, adjust_val):
        B = st_batch.B
        device = st_batch.get_device()
        n_nodes = st_batch.gs.max_problem_dim
        assert adjust_val.shape == (B, 1, 3) # CoM of delxt
        delxt_CoM = adjust_val

        xt_pos = st_batch.tuple_batch[0]
        assert xt_pos.shape == (B, n_nodes, 3)
        atom_mask = torch.arange(n_nodes).view(1, -1) < st_batch.get_dims().view(-1, 1) # (B, n_nodes)
        assert atom_mask.shape == (B, n_nodes)
        atom_mask = atom_mask.long().to(device)
        node_mask = atom_mask.unsqueeze(2)
        assert node_mask.shape == (B, n_nodes, 1)

        xt_pos_from_y = (xt_pos - delxt_CoM) * node_mask

        assert xt_pos_from_y.shape == (B, n_nodes, 3)

        auto_target = torch.cat([
            xt_pos_from_y.flatten(start_dim=1),
            st_batch.tuple_batch[1].flatten(start_dim=1),
            st_batch.tuple_batch[2].flatten(start_dim=1)
        ], dim=1)
        assert auto_target.shape == (B, n_nodes * (3+5+1))

        return auto_target

    def get_nearest_atom(self, st_batch, delxt_st_batch):
        # assumes we are doing final dim deletion
        B = st_batch.B
        device = st_batch.get_device()

        x_full = st_batch.tuple_batch[0] # (B, n_nodes, 3)
        full_dims = st_batch.get_dims() # (B,)
        x_del = delxt_st_batch.tuple_batch[0] # (B, n_nodes, 3)

        # if full dim is 3 then x_full is [0, 1, 2] so missing atom is at idx 2

        missing_atom_pos = x_full[torch.arange(B, device=device).long(), (full_dims - 1).long(), :] # (B, 3)

        distances_to_missing = torch.sum((x_del - missing_atom_pos.unsqueeze(1)) ** 2, dim=2) # (B, n_nodes)

        atom_mask = torch.arange(st_batch.gs.max_problem_dim).view(1, -1) < delxt_st_batch.get_dims().view(-1, 1) # (B, n_nodes)
        atom_mask = atom_mask.to(device).long()

        distances_to_missing = atom_mask * distances_to_missing + (1-atom_mask) * 1e3

        nearest_atom = torch.argmin(distances_to_missing, dim=1) # (B,)

        return nearest_atom
