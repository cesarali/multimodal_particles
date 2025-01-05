import os
import pytest
from pprint import pprint
from multimodal_particles.data.particle_clouds.dataloader import MultimodalBridgeDataloaderModule
from multimodal_particles.data.particle_clouds.jets import JetDataclass
from multimodal_particles.config_classes.multimodal_bridge_matching_config import MultimodalBridgeMatchingConfig
from multimodal_particles.models.generative.multimodal_bridge_matching import MultiModalBridgeMatching
from multimodal_particles.utils.experiment_configs import namespace_to_dict,dict_to_yaml
from multimodal_particles import test_resources_dir


def test_configs():
    #config_file_path=os.path.join(test_resources_dir, "configs_files", "config-mbm-test-new.yaml")
    config_file_path = os.path.join(test_resources_dir, "configs_files", "config-mbm-test.yaml")
    model_config = MultimodalBridgeMatchingConfig.from_yaml(config_file_path)
    assert model_config is not None

def test_random_databatch():
    #config_file_path=os.path.join(test_resources_dir, "configs_files", "config-mbm-test-new.yaml")
    config_file_path=os.path.join(test_resources_dir, "configs_files", "config-mbm-test.yaml")
    model_config = MultimodalBridgeMatchingConfig.from_yaml(config_file_path)
    random_databatch = MultimodalBridgeDataloaderModule.random_databatch(model_config)

    # create datamodule
    jets = JetDataclass(config=model_config)
    jets.preprocess()
    dataloader = MultimodalBridgeDataloaderModule(config=model_config, dataclass=jets)
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
    #config_file_path=os.path.join(test_resources_dir, "configs_files", "config-mbm-test-new.yaml")
    config_file_path=os.path.join(test_resources_dir, "configs_files", "config-mbm-test.yaml")
    model_config = MultimodalBridgeMatchingConfig.from_yaml(config_file_path)
    random_databatch = MultimodalBridgeDataloaderModule.random_databatch(model_config)
    model = MultiModalBridgeMatching(model_config)
    state = model.sample_bridges(random_databatch)
    print(state.time.shape, state.continuous.shape, state.discrete.shape, state.absorbing.shape)


if __name__=="__main__":
    test_model()