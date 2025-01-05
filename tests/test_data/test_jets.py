import os
import pytest
from pprint import pprint
from multimodal_particles.utils.experiment_configs import load_config
from multimodal_particles.data.particle_clouds.dataloader import MultimodalBridgeDataloaderModule
from multimodal_particles.data.particle_clouds.jets import JetDataclass
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig
from multimodal_particles.models.generative.multimodal_bridge_matching import MultiModalBridgeMatching
from multimodal_particles.utils.experiment_configs import namespace_to_dict,dict_to_yaml
from multimodal_particles import test_resources_dir

def test_configs():
    config_file_path = os.path.join(test_resources_dir, "configs_files", "config-mbm-test.yaml")
    model_config = MultimodalBridgeMatchingConfig.from_yaml(config_file_path)
    assert model_config is not None

def test_databatch():
    #obtain configs
    config_file_path = os.path.join(test_resources_dir, "configs_files", "config-mbm-test.yaml")
    model_config = MultimodalBridgeMatchingConfig.from_yaml(config_file_path)
    
    # create datamodule
    jets = JetDataclass(config=model_config)
    jets.preprocess()
    dataloader = MultimodalBridgeDataloaderModule(config=model_config, dataclass=jets)
    
    databatch = next(dataloader.train.__iter__())
    assert databatch is not None
    
if __name__=="__main__":
    test_databatch()