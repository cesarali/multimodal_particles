import os
import pytest
import torch
from multimodal_particles.models import AbsorbingFlow
from multimodal_particles.config_classes.absorbing_flows_config import AbsorbingConfig
from multimodal_particles.data.particle_clouds.jets import JetDataclass
from multimodal_particles.data.particle_clouds.jets_dataloader import JetsDataloaderModule
from multimodal_particles.models.generative.bridges import AbsorbingBridge
from multimodal_particles import test_resources_dir

def test_config():
    config_file_path = os.path.join(test_resources_dir, "configs_files", "config-absorbing-test.yaml")
    config = AbsorbingConfig()
    config.to_yaml(config_file_path)
    config = AbsorbingConfig.from_yaml(config_file_path)


def test_bridge():
    config = AbsorbingConfig()
    model = AbsorbingFlow(config)
    random_databatch = JetsDataloaderModule.random_databatch(config)

    absorbing_bridge = AbsorbingBridge(config)

    # all equal to target at time 1
    t = torch.ones(
            random_databatch.target_continuous.shape[0], device=random_databatch.target_continuous.device
    ).type_as(random_databatch.target_continuous)
    time = model.reshape_time(t, random_databatch.target_continuous)

    mask_t = absorbing_bridge.sample(time,random_databatch.target_mask)
    assert (mask_t == random_databatch.target_mask).all()

    # all existing at time 0
    t = torch.zeros(
            random_databatch.target_continuous.shape[0], device=random_databatch.target_continuous.device
    ).type_as(random_databatch.target_continuous)
    time = model.reshape_time(t, random_databatch.target_continuous)

    mask_t = absorbing_bridge.sample(time,random_databatch.target_mask)
    assert (mask_t == 1).all()

    # sample full state
    state = model.sample_bridges(random_databatch)
    assert state is not None

def test_absorbing_head():
    config = AbsorbingConfig()
    model = AbsorbingFlow(config)

    jets = JetDataclass(config=config)
    jets.preprocess()
    random_databatch = JetsDataloaderModule.random_databatch(config)
    #databatch = next(dataloader.train.__iter__())

    state = model.sample_bridges(random_databatch)
    forward_head = model(state,random_databatch)
    loss_absorbing = model.loss_absorbing(forward_head,random_databatch)
    assert loss_absorbing is not None


if __name__=="__main__":
    test_config()