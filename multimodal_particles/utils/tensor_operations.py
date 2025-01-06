import torch

def create_and_apply_mask_3(one_tensor_from_databatch,new_dims_dev,device):
    """ creates and apply mask for shape of len 3 for graphical structure objects """
    one_tensor_mask = torch.arange(one_tensor_from_databatch.shape[1], device=device).view(1, -1, 1).repeat(one_tensor_from_databatch.shape[0], 1, one_tensor_from_databatch.shape[2])
    one_tensor_mask = (one_tensor_mask < new_dims_dev.view(-1, 1, 1))
    one_tensor_from_databatch = one_tensor_from_databatch * one_tensor_mask
    return one_tensor_from_databatch,one_tensor_mask

def create_and_apply_mask_2(one_tensor_from_databatch,new_dims_dev,device):
    """ creates and apply mask for shape of len 2 for graphical structure objects """
    one_tensor_mask = torch.arange(one_tensor_from_databatch.shape[1], device=device).view(1, -1).repeat(one_tensor_from_databatch.shape[0], 1)
    one_tensor_mask = (one_tensor_mask < new_dims_dev.view(-1, 1))
    one_tensor_from_databatch = one_tensor_from_databatch * one_tensor_mask
    return one_tensor_from_databatch,one_tensor_mask