import os
import pytest
from pprint import pprint
from multimodal_particles.config_classes.transdimensional_config_unconditional import TransdimensionalEpicConfig

def test_config():
    from multimodal_particles import config_dir
    config = TransdimensionalEpicConfig()
    config_path = os.path.join(config_dir,"config-transepic-berlin.yaml")
    config.to_yaml(config_path)
    config_read = TransdimensionalEpicConfig.from_yaml(config_path)
    assert config_read is not None

if __name__=="__main__":
    test_config()