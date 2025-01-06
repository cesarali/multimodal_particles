import os
import torch
import pytest
from pprint import pprint
from multimodal_particles import config_dir

from multimodal_particles.data.particle_clouds.utils import sizes_to_histograms
from multimodal_particles.utils.experiment_configs import load_config
from multimodal_particles.models.generative.transdimensional import TransdimensionalJumpDiffusion
from multimodal_particles.config_classes.transdimensional_config_unconditional import TransdimensionalEpicConfig
from multimodal_particles.data.particle_clouds.dataloader import JetsGraphicalStructure
from multimodal_particles.models.generative.transdimensional.structure import (
    Structure,
    StructuredDataBatch
)
from multimodal_particles.data.particle_clouds.dataloader import MultimodalBridgeDataloaderModule
from multimodal_particles.data.particle_clouds.jets import JetDataclass

def test_config():
    config_path = os.path.join(config_dir,"config-transepic-berlin.yaml")
    config = TransdimensionalEpicConfig()
    config.to_yaml(config_path)
    config_read = TransdimensionalEpicConfig.from_yaml(config_path)
    assert config_read is not None

def test_graphical_structure():
    from multimodal_particles.data.particle_clouds.utils import sizes_to_histograms
    from multimodal_particles import test_resources_dir

    #obtain configs
    config = TransdimensionalEpicConfig()
    config.data.return_type = "list"

    # create datamodule
    jets = JetDataclass(config=config)
    jets.preprocess()
    datamodule = MultimodalBridgeDataloaderModule(config=config, jetdataset=jets)
    dims, *data = next(datamodule.train.__iter__())

    # create module
    model = TransdimensionalJumpDiffusion(config,datamodule)

    # create structures
    graphical_structure = JetsGraphicalStructure(datamodule)
    st_batch = StructuredDataBatch(data,
                                   dims,
                                   datamodule.observed,
                                   datamodule.exists,
                                   datamodule.is_onehot,
                                   datamodule.graphical_structure)
    
    # checks that all shapes as defined from the
    print(graphical_structure.shapes_with_onehot())
    print(data[0].shape,data[1].shape)
    B = data[0].size(0)
    for shapes_index,shapes_from_graphical_structure in enumerate(graphical_structure.shapes_with_onehot()):
        assert data[shapes_index].shape == (B, *shapes_from_graphical_structure)

    device = st_batch.get_device()
    x0_dims = st_batch.get_dims()

    ts = config.loss_kwargs.min_t + (1-config.loss_kwargs.min_t) * torch.rand((B,)) # (B,)

    # delete some dimensions
    dims_xt = model.forward_rate.get_dims_at_t(
            start_dims=st_batch.get_dims(),
            ts =ts
        ).int() # (B,)
    
    assert dims_xt is not None
    
if __name__=="__main__":
    test_graphical_structure()