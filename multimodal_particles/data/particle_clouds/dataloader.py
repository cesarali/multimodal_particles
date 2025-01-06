import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from collections import namedtuple
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig
from multimodal_particles.data.transdimensional_base import GraphicalStructureBase
from multimodal_particles.models.architectures.egnn_utils import DistributionNodes
from multimodal_particles.data.particle_clouds.utils import sizes_to_histograms
from multimodal_particles.data.particle_clouds.jets import JetDataclass

from multimodal_particles.utils.tensor_operations import (
    create_and_apply_mask_2,
    create_and_apply_mask_3
)

class MultimodalBridgeDataset(Dataset):

    def __init__(self, data:JetDataclass, return_type='namedtuple'):
        """
        Initialize the dataset.

        Args:
            data: The data object containing source, target, and context attributes.
            return_type: Specifies the return type for __getitem__.
                         Options are 'namedtuple' or 'list'. Default is 'namedtuple'.
        """
        self.data = data
        self.attributes = []
        self.return_type = return_type
        self.vocab_size_features = data.vocab_size_features
        self.vocab_size_context = data.vocab_size_context
        self.return_type = self.data.config.data.return_type

        # ...source
        if hasattr(self.data.source, "continuous") and self.return_type == "namedtuple":
            self.attributes.append("source_continuous")
            self.source_continuous = self.data.source.continuous

        if hasattr(self.data.source, "discrete") and self.return_type == "namedtuple":
            self.attributes.append("source_discrete")
            self.source_discrete = self.data.source.discrete

        if hasattr(self.data.source, "mask") and self.return_type == "namedtuple":
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
            if self.return_type != "list":
                self.attributes.append("target_mask")
            self.target_mask = self.data.target.mask

        # ...context
        if hasattr(self.data, "context_continuous"):
            self.attributes.append("context_continuous")
            self.context_continuous = self.data.context_continuous

        if hasattr(self.data, "context_discrete"):
            self.attributes.append("context_discrete")
            self.context_discrete = self.data.context_discrete

        self.databatch_namedtuple = namedtuple("databatch", self.attributes)

    def __getitem__(self, idx):
        """
        Retrieve a data item by index.

        Args:
            idx: Index of the data item.

        Returns:
            if return_type == namedtuple
                A namedtuple or list of tensors based on the return_type.
            else return_type == list
                A list with the first element equal to the number of particles
        """
        if self.return_type == 'namedtuple':
            data = [getattr(self, attr)[idx] for attr in self.attributes]
            return self.databatch_namedtuple(*data)
        elif self.return_type == 'list':
            target_mask = getattr(self, "target_mask")[idx]
            n_particles = target_mask.squeeze().sum(axis=-1)
            data = [n_particles]
            for attr in self.attributes:
                if attr != "target_mask":
                    value = getattr(self, attr)[idx]
                    if attr in ['source_discrete', 'target_discrete']:
                        # Apply one-hot encoding for discrete data
                        n_classes = self.vocab_size_features
                        one_hot = np.eye(n_classes)[value]  # Convert to one-hot encoding
                        value = torch.tensor(one_hot, dtype=torch.float32)  # Convert to tensor
                        value = value.squeeze()
                    if attr == "context_discrete":
                        n_classes = self.vocab_size_context
                        one_hot = np.eye(n_classes)[value]  # Convert to one-hot encoding
                        value = torch.tensor(one_hot, dtype=torch.float32)  # Convert to tensor
                        value = value.squeeze()
                    data.append(value)
            return data
        else:
            raise ValueError("Invalid return_type. Choose 'namedtuple' or 'list'.")

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            Length of the dataset (number of items).
        """
        return len(self.data.target)

    def __iter__(self):
        """
        Iterate over the dataset.

        Yields:
            Items of the dataset one by one.
        """
        for idx in range(len(self)):
            yield self[idx]

    def get_available_keys(self):
        """
        Get the names of keys (attributes) available in the dataset.

        Returns:
            List of attribute names.
        """
        return self.attributes

class MultimodalBridgeDataloaderModule:
    
    def __init__(
        self, 
        config:MultimodalBridgeMatchingConfig, 
        jetdataset, 
        batch_size: int = None, 
        data_split_frac: tuple = None
    ):
        self.dataclass = jetdataset
        self.config = config
        self.dataset = MultimodalBridgeDataset(jetdataset,return_type=config.data.return_type)

        # sets metadata after reading data
        self.histogram_target = sizes_to_histograms(self.dataset.target_mask.squeeze().sum(axis=1))
        if self.config.data.return_type == "namedtuple":
            self.histogram_source = sizes_to_histograms(self.dataset.source_mask.squeeze().sum(axis=1))

        self.data_split = (
            self.config.data.data_split_frac
            if data_split_frac is None
            else data_split_frac
        )
        self.batch_size = (
            self.config.data.batch_size if batch_size is None else batch_size
        )
        self.set_dataloader()

        # graphical structure is an object that allows the destruction and creation of particles 
        # as defined in the transdimensional code is only necesary for transdimensional classes
        if hasattr(self.config.data,"graphical_structure"):
            self.set_batch_handlers()
            self.graphical_structure = JetsGraphicalStructure(self)

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

    def set_dataloader(self):
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
    def random_databatch(config:MultimodalBridgeMatchingConfig):
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
        batch_size = config.data.batch_size
        max_num_particles = config.data.max_num_particles
        dim_continuous = config.data.dim_features_continuous
        dim_discrete = config.data.dim_features_discrete
        vocab_size = config.data.vocab_size_features

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
    
    def update_config(self,model_config:MultimodalBridgeMatchingConfig):
        model_config.data.target_info["hist_num_particles"] = self.histogram_target
        if self.config.data.return_type == "namedtuple":
            model_config.data.source_info["hist_num_particles"] = self.histogram_source
        return model_config
    
    def set_without_onehot_shapes(self,names_in_batch):
        config = self.config
        max_num_particles = config.data.max_num_particles
        without_onehot_shapes = []
        for name_index, name in enumerate(names_in_batch):
            if "target_continuous" == name:
                without_onehot_shapes.append(torch.Size([max_num_particles, config.data.dim_features_continuous]))
            if "target_mask" == name:
                without_onehot_shapes.append(torch.Size([max_num_particles, 1]))
            if "context_continuous" == name:
                without_onehot_shapes.append(torch.Size([max_num_particles, config.data.dim_context_continuous]))
            if "context_discrete" == name:
                without_onehot_shapes.append(torch.Size([max_num_particles, config.data.vocab_size_features]))
        self.without_onehot_shapes = without_onehot_shapes

    def set_onehot_shapes(self,names_in_batch):
        config = self.config
        max_num_particles = config.data.max_num_particles
        with_onehot_shapes = []
        for name_index, name in enumerate(names_in_batch):
            if "target_continuous" == name:
                with_onehot_shapes.append(torch.Size([max_num_particles, config.data.dim_features_continuous]))
            if "target_discrete" == name:
                with_onehot_shapes.append(torch.Size([max_num_particles, config.data.vocab_size_features]))
            if "target_mask" == name:
                with_onehot_shapes.append(torch.Size([max_num_particles, 1]))
            if "context_continuous" == name:
                with_onehot_shapes.append(torch.Size([max_num_particles, config.data.dim_context_continuous]))
            if "context_discrete" == name:
                with_onehot_shapes.append(torch.Size([max_num_particles, config.data.vocab_size_features]))
        self.with_onehot_shapes = with_onehot_shapes

    def set_batch_handlers(self):
        """
        For the transdimensional structure classes, one needs to known which elements of the
        batch list are observed and one hot, as well as sizes and names
        """
        names_in_batch = self.dataset.get_available_keys()
        self.names_in_batch = names_in_batch
        self.observed = np.zeros(len(names_in_batch)).astype(int)
        self.is_onehot = np.zeros(len(names_in_batch)).astype(int)
        self.exists = np.ones(len(names_in_batch)).astype(int)
        self.name_to_index = dict(zip(names_in_batch,range(len(names_in_batch))))

        if "target_discrete" in names_in_batch:
            self.is_onehot[self.name_to_index["target_discrete"]] = 1
        
        if "context_continuous" in names_in_batch:
            self.observed[self.name_to_index["context_continuous"]] = 1

        if "context_discrete" in names_in_batch:
            self.observed[self.name_to_index["context_discrete"]] = 1

        self.set_onehot_shapes(names_in_batch)
        self.set_without_onehot_shapes(names_in_batch)

class JetsGraphicalStructure(GraphicalStructureBase):
    
    def __init__(
            self,
            datamodule:MultimodalBridgeDataloaderModule,
        ):
        config = datamodule.config
        histogram = datamodule.histogram_target

        self.names_in_batch = datamodule.names_in_batch
        self.max_num_particles = config.data.max_num_particles
        self.num_jets = config.data.num_jets

        # dimensions
        self.dim_features_continuous = config.data.dim_features_continuous
        self.dim_features_discrete = config.data.dim_features_discrete
        self.dim_context_continuous = config.data.dim_context_continuous
        self.dim_context_discrete = config.data.dim_context_discrete
        self.vocab_size_features = config.data.vocab_size_features
        self.vocab_size_context = config.data.vocab_size_context

        self.with_onehot_shapes = datamodule.with_onehot_shapes
        self.without_onehot_shapes = datamodule.without_onehot_shapes

        self.nodes_dist = DistributionNodes(histogram)

    def shapes_without_onehot(self):
        """
        Returns the shapes of the databatches without one-hot encoding dynamically based on attributes.

        Returns:
            List[torch.Size]: Shapes of the data tensors without one-hot encoding.
        """
        return self.without_onehot_shapes

    def shapes_with_onehot(self):
        """
        Returns the shapes of the databatches with one-hot encoding dynamically based on attributes.

        Returns:
            List[torch.Size]: Shapes of the data tensors with one-hot encoding.
        """
        return self.with_onehot_shapes

    def remove_problem_dims(self, data, new_dims,name,name_to_index):
        # pos, atom_type, charge, alpha, homo, lumo, gap, mu, Cv = data

        #B = pos.shape[0]
        #assert atom_type.shape == (B, *self.shapes_with_onehot()[1])
        #assert charge.shape == (B, *self.shapes_with_onehot()[2])

        device = data[0].device
        new_dims_dev = new_dims.to(device)

        databatch_with_dimensions_removed = []
        for name_index, name in enumerate(self.names_in_batch):
            if "target_continuous" == name:
                tensor_index = name_to_index["target_continuous"]
                one_tensor_from_databatch = data[tensor_index]
                B = one_tensor_from_databatch.size(0)
                new_tensor = create_and_apply_mask_3(one_tensor_from_databatch,new_dims_dev,device)
                databatch_with_dimensions_removed.append(new_tensor)
                #assert pos.shape == (B, *self.shapes_with_onehot()[0])
            if "target_discrete" == name:
                tensor_index = name_to_index["target_continuous"]
                one_tensor_from_databatch = data[tensor_index]
                B = one_tensor_from_databatch.size(0)
                new_tensor = create_and_apply_mask_3(one_tensor_from_databatch,new_dims_dev,device)
                databatch_with_dimensions_removed.append(new_tensor)        
            if "target_mask" == name:
                tensor_index = name_to_index["target_continuous"]
                one_tensor_from_databatch = data[tensor_index]
                B = one_tensor_from_databatch.size(0)
                new_tensor = create_and_apply_mask_3(one_tensor_from_databatch,new_dims_dev,device)
                databatch_with_dimensions_removed.append(new_tensor)        
            if "context_continuous" == name:
                tensor_index = name_to_index["target_continuous"]
                one_tensor_from_databatch = data[tensor_index]
                B = one_tensor_from_databatch.size(0)
                new_tensor = create_and_apply_mask_3(one_tensor_from_databatch,new_dims_dev,device)
                databatch_with_dimensions_removed.append(new_tensor)
            if "context_discrete" == name:
                tensor_index = name_to_index["target_continuous"]
                one_tensor_from_databatch = data[tensor_index]
                B = one_tensor_from_databatch.size(0)
                new_tensor = create_and_apply_mask_3(one_tensor_from_databatch,new_dims_dev,device)
                databatch_with_dimensions_removed.append(new_tensor)

        return databatch_with_dimensions_removed

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
