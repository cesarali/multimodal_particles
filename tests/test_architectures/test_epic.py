import torch


def test_epic():
    from utils.experiment_configs import Configs
    from data.particle_clouds.jets import JetDataclass

    config_path = r"/home/cesarali/Codes/multimodal_particles/configs_files/config-berlin.yaml"

    config = Configs(config_path)
    jets = JetDataclass(config=config)
    print(jets.source.continuous.shape, jets.source.discrete.shape, jets.source.mask.shape)
    

if __name__=="__main__":
    test_epic()