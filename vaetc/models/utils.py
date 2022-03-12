import torch

def detach_dict(obj):
    """ Detach scalar tensors in a :class:`dict`.

    Args:
        obj (:class:`dict` of :class:`torch.Tensor`): a dict containing scalar tensors to be detached

    Returns:
        :class:`dict` of :class:`torch.Tensor`: detached scalars
    """
    
    detached = {}
    
    for key in obj:
        
        value = obj[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        
        detached[key] = float(value)
    
    return detached