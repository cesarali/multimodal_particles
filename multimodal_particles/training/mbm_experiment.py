import yaml
from multimodal_particles.training.basic_experiments import BasicLightningExperiment
from multimodal_particles.utils.dataloader import DataloaderModule 
from multimodal_particles.utils.experiment_configs import Configs
from multimodal_particles.data.particle_clouds.jets import JetDataclass
from multimodal_particles.models.generative.multimodal_bridge_matching import MultiModalBridgeMatching


class MultimodalBridgeMatchingExperiment(BasicLightningExperiment):

    def __init__(self, config:Configs = None, experiment_dir = None, map_location = "cuda"):
        super().__init__(config, experiment_dir, map_location)

    def setup_experiment_from_dir(self, experiment_dir):
        pass

    def setup_datamodule(self):
        jets = JetDataclass(config=self.config)
        jets.preprocess()
        self.datamodule = DataloaderModule(config=self.config, dataclass=jets)

    def setup_model(self):
        self.model = MultiModalBridgeMatching(self.config)

    def save_test_samples(self):
        pass

    def save_hyperparameters_to_yaml(self, hyperparams, file_path: str):
        """
        Saves hyperparameters to a YAML file.
        """
        with open(file_path, "w") as yaml_file:
            yaml.dump(hyperparams.config_dict, yaml_file)
    
