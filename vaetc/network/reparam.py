import torch

def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """ The reparameterization trick (https://arxiv.org/abs/1312.6114) in a Gaussian distribution.

    Args:
        mean (torch.Tensor): The mean of the distribution.
        logvar (torch.Tensor): The log-variance of the distribution.

    Returns:
        torch.Tensor: sampled values
    """
    
    std = torch.exp(logvar * 0.5)
    eps = torch.randn_like(std)

    return mean + std * eps