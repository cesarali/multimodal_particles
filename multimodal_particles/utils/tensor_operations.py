import torch

def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

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

def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected

def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)
