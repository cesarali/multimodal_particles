import os
import pytest
from pprint import pprint
from multimodal_particles import config_dir

from multimodal_particles.data.particle_clouds.utils import sizes_to_histograms
from multimodal_particles.utils.experiment_configs import load_config
from multimodal_particles.models.generative.transdimensional import TransdimensionalJumpDiffusion
from multimodal_particles.config_classes.transdimensional_config_unconditional import TransdimensionalEpicConfig
from multimodal_particles.data.particle_clouds.dataloader import JetsGraphicalStructure
from multimodal_particles.models.generative.transdimensional.structure import Structure
from multimodal_particles.data.particle_clouds.jets import JetDataclass
from multimodal_particles.data.particle_clouds.dataloader import MultimodalBridgeDataloaderModule

def test_config():
    config = TransdimensionalEpicConfig()
    config_path = os.path.join(config_dir,"config-transepic-berlin.yaml")
    config.to_yaml(config_path)
    config_read = TransdimensionalEpicConfig.from_yaml(config_path)
    assert config_read is not None

def test_structure():
    from multimodal_particles.data.particle_clouds.utils import sizes_to_histograms
    from multimodal_particles import test_resources_dir

    #obtain configs
    config_file_path = os.path.join(test_resources_dir, "configs_files", "config-mbm-test.yaml")
    full_config = load_config(config_file_path)
    # create datamodule
    jets = JetDataclass(config=full_config)
    jets.preprocess()
    dataloader = MultimodalBridgeDataloaderModule(config=full_config, dataclass=jets)
    
    config = TransdimensionalEpicConfig()
    histogram = sizes_to_histograms(jets.target.mask.squeeze().sum(axis=1))
    graphical_structure  = JetsGraphicalStructure(config.max_num_particles,histogram)
    assert graphical_structure is not None


if __name__=="__main__":
    test_structure()