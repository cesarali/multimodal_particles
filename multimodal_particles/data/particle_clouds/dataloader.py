import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from collections import namedtuple
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig

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
            self.config.train.data_split_frac
            if data_split_frac is None
            else data_split_frac
        )
        self.batch_size = (
            self.config.train.batch_size if batch_size is None else batch_size
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
        batch_size = config.train.batch_size
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
    
    @staticmethod
    def update_model_config(full_config,model_config:MultimodalBridgeMatchingConfig):        
        model_config.dim_features_continuous = full_config.data.dim.features_continuous
        model_config.dim_features_discrete = full_config.data.dim.features_discrete
        model_config.dim_context_continuous = full_config.data.dim.context_continuous
        model_config.dim_context_discrete = full_config.data.dim.context_discrete

        model_config.vocab_size_features = full_config.data.vocab_size.features
        model_config.vocab_size_context = full_config.data.vocab_size.context
        return model_config