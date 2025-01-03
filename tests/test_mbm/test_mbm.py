import os
import pytest
from pprint import pprint
from multimodal_particles.utils.experiment_configs import load_config
from multimodal_particles.data.particle_clouds.dataloader import MultimodalBridgeDataloaderModule
from multimodal_particles.data.particle_clouds.jets import JetDataclass
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig
from multimodal_particles.models.generative.multimodal_bridge_matching import MultiModalBridgeMatching
from multimodal_particles.utils.experiment_configs import namespace_to_dict,dict_to_yaml

def test_configs():
    config_file_path = r"/home/cesarali/Codes/multimodal_particles/configs_files/config-berlin.yaml"
    full_config = load_config(config_file_path)
    model_config = MultimodalBridgeMatchingConfig.from_full_config(full_config)
    assert model_config is not None

def test_random_databatch():
    #obtain configs
    config_file_path = r"/home/cesarali/Codes/multimodal_particles/configs_files/config-berlin.yaml"
    full_config = load_config(config_file_path)
    
    #rand
    random_databatch = MultimodalBridgeDataloaderModule.random_databatch(full_config)

    # create datamodule
    jets = JetDataclass(config=full_config)
    jets.preprocess()
    dataloader = MultimodalBridgeDataloaderModule(config=full_config, dataclass=jets)
    databatch = next(dataloader.train.__iter__())

    # Check that all fields have the same shape
    for field in random_databatch._fields:
        random_tensor_shape = getattr(random_databatch, field).shape
        data_tensor_shape = getattr(databatch, field).shape
        
        assert random_tensor_shape == data_tensor_shape, (
            f"Shape mismatch in field '{field}': "
            f"{random_tensor_shape} (random) vs {data_tensor_shape} (data)"
        )

    print("All fields have matching shapes.")

def test_model():
    config_file_path = r"/home/cesarali/Codes/multimodal_particles/configs_files/config-berlin.yaml"
    full_config = load_config(config_file_path)
    model_config = MultimodalBridgeMatchingConfig.from_full_config(full_config)
    model_config = MultimodalBridgeDataloaderModule.update_model_config(full_config,model_config)
    model = MultiModalBridgeMatching(model_config)
    random_databatch = MultimodalBridgeDataloaderModule.random_databatch(full_config)
    state = model.sample_bridges(random_databatch)

if __name__=="__main__":
    test_random_databatch()
