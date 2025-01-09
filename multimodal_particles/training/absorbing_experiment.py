import yaml

from multimodal_particles.models import AbsorbingFlow
from multimodal_particles.data.particle_clouds.jets import JetDataclass
from multimodal_particles.data.particle_clouds.jets_dataloader import JetsDataloaderModule 
from multimodal_particles.config_classes.absorbing_flows_config import AbsorbingConfig

from multimodal_particles.training.basic_experiments import BasicLightningExperiment
from multimodal_particles.utils.experiment_configs import namespace_to_dict,dict_to_yaml

class AbsorbingExperiment(BasicLightningExperiment):

    def __init__(self, config:AbsorbingConfig = None, experiment_dir = None, map_location = "cuda"):
        super().__init__(config, experiment_dir, map_location)

    def setup_experiment_from_dir(self, experiment_dir):
        pass

    def setup_datamodule(self):
        jets = JetDataclass(config=self.config)
        jets.preprocess()
        self.datamodule = JetsDataloaderModule(config=self.config, jetdataset=jets)

    def setup_model(self):
        self.model_config = self.datamodule.update_config(self.config)
        self.model = AbsorbingFlow(self.model_config)        
    
    def save_test_samples(self):
        pass

    def save_hyperparameters_to_yaml(self, hyperparams, file_path: str):
        """
        Saves hyperparameters to a YAML file.
        """
        with open(file_path, "w") as yaml_file:
            config_dict = namespace_to_dict(self.config)
            yaml.dump(config_dict, yaml_file)
    
