from multimodal_particles.training.mbm_experiment import MultimodalBridgeMatchingExperiment
from multimodal_particles.utils.experiment_configs import Configs

def train_mbm(config):
    experiment = MultimodalBridgeMatchingExperiment(config=config)
    experiment.train()

if __name__=="__main__":
    config_file_path = r"/home/cesarali/Codes/multimodal_particles/configs_files/config-berlin.yaml"
    config = Configs(config_file_path)
    train_mbm(config)