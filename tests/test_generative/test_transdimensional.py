import os
import torch
import pytest
from pprint import pprint
from multimodal_particles import config_dir

from multimodal_particles.models.generative.transdimensional import TransdimensionalJumpDiffusion
from multimodal_particles.config_classes.transdimensional_config_unconditional import TransdimensionalEpicConfig
from multimodal_particles.data.particle_clouds.jets_dataloader import JetsGraphicalStructure
from multimodal_particles.models.generative.transdimensional.structure import (
    Structure,
    StructuredDataBatch
)
from multimodal_particles.data.particle_clouds.jets_dataloader import JetsDataloaderModule
from multimodal_particles.data.particle_clouds.jets import JetDataclass

def test_config():
    config_path = os.path.join(config_dir,"config-transepic-berlin.yaml")
    config = TransdimensionalEpicConfig()
    config.to_yaml(config_path)
    config_read = TransdimensionalEpicConfig.from_yaml(config_path)
    assert config_read is not None

def test_graphical_structure():
    #obtain configs
    config = TransdimensionalEpicConfig()
    config.data.return_type = "list"

    # create datamodule
    jets = JetDataclass(config=config)
    jets.preprocess()
    datamodule = JetsDataloaderModule(config=config, jetdataset=jets)
    dims, *data = next(datamodule.train.__iter__())

    # create module
    model = TransdimensionalJumpDiffusion(config,datamodule)

    # create structures
    structure = Structure(datamodule.exist, datamodule.observed, datamodule)
    graphical_structure = JetsGraphicalStructure(datamodule)
    st_batch = StructuredDataBatch(data,
                                   dims,
                                   datamodule.observed,
                                   datamodule.exist,
                                   datamodule.is_onehot,
                                   datamodule.graphical_structure)
    
    # checks that all shapes as defined from the
    print(graphical_structure.shapes_with_onehot())
    print(data[0].shape,data[1].shape)
    print(dims)

    B = data[0].size(0)
    for shapes_index,shapes_from_graphical_structure in enumerate(graphical_structure.shapes_with_onehot()):
        assert data[shapes_index].shape == (B, *shapes_from_graphical_structure)

    ts = config.loss_kwargs.min_t + (1-config.loss_kwargs.min_t) * torch.rand((B,)) # (B,)
    # delete some dimensions
    device = st_batch.get_device()
    x0_dims = st_batch.get_dims()

    dims_xt = model.forward_rate.get_dims_at_t(
            start_dims=st_batch.get_dims(),
            ts =ts
        ).int() # (B,)
    
    assert dims_xt is not None
    st_batch.delete_dims(new_dims=dims_xt)
    st_batch.gs.adjust_st_batch(st_batch)
    x, y = st_batch.get_flat_lats_and_obs()

    graphical_structure.adjust_st_batch(st_batch)
    mean, std = model.noise_schedule.get_p0t_stats(st_batch, ts.to(device))

def test_model():
    from multimodal_particles.models.generative.transdimensional.loss import add_noise

    #obtain configs
    config = TransdimensionalEpicConfig()
    config.data.return_type = "list"

    # create datamodule
    jets = JetDataclass(config=config)
    jets.preprocess()
    datamodule = JetsDataloaderModule(config=config, jetdataset=jets)
    dims, *data = next(datamodule.train.__iter__())

    # create module
    model = TransdimensionalJumpDiffusion(config,datamodule)
    st_batch = StructuredDataBatch(data,
                                   dims,
                                   datamodule.observed,
                                   datamodule.exist,
                                   datamodule.is_onehot,
                                   datamodule.graphical_structure)

    # inputs network and structured data batch
    st_batch,ts,x0_dims,B,noise,device,x = add_noise(st_batch,model.noise_schedule,model.forward_rate,model.jump_diffusion_loss.min_t)

    # first network pass
    to_predict = {'eps': 'eps', 'x0': 'x0', 'edm': 'x0'}[model.jump_diffusion_loss.loss_type]
    if model.jump_diffusion_loss.nearest_atom_pred:
        D_xt, rate_xt, dummy_mean_std, x0_dim_logits, _ = model.net(
            st_batch, ts=ts.to(device), forward_rate=model.forward_rate,
            predict=to_predict, nearest_atom=torch.zeros((B,), device=device).long()
        )
    else:
        D_xt, rate_xt, dummy_mean_std, x0_dim_logits = model.net(st_batch, ts=ts.to(device), forward_rate=model.forward_rate, predict=to_predict)
    assert rate_xt.shape == (B, 1)

if __name__=="__main__":
    test_model()