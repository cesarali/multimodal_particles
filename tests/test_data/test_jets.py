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

@pytest.mark.skip(reason="Skipping this test since it loads data")
def test_databatch():
    #obtain configs
    config_file_path = os.path.join(test_resources_dir, "configs_files", "config-mbm-test.yaml")
    full_config = load_config(config_file_path)
    # create datamodule
    jets = JetDataclass(config=full_config)
    jets.preprocess()
    dataloader = MultimodalBridgeDataloaderModule(config=full_config, dataclass=jets)
    
    databatch = next(dataloader.train.__iter__())
    assert databatch is not None
    
if __name__=="__main__":
    test_databatch()