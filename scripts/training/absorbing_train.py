import os
from multimodal_particles import test_resources_dir
from multimodal_particles.training.absorbing_experiment import AbsorbingExperiment
from multimodal_particles.config_classes.absorbing_flows_config import AbsorbingConfig

def train_absorbing(config):
    experiment = AbsorbingExperiment(config=config)
    experiment.train()

if __name__=="__main__":
    config_file_path = os.path.join(test_resources_dir, "configs_files", "config-absorbing-test.yaml")
    config = AbsorbingConfig.from_yaml(config_file_path)
    train_absorbing(config)
