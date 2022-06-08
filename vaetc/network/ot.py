import math
import torch

def sinkhorn(
    a: torch.Tensor, b: torch.Tensor, cost_matrix: torch.Tensor,
    num_iterations: int = 50, eps: float = 0.01
) -> torch.Tensor:
    """ Sinkhorn-Knopp Algorithm

    Args:
        a (torch.Tensor): source distribution
        b (torch.Tensor): drain distribution
        cost_matrix (torch.Tensor): cost matrix; the element cost_matrix[i,j] corresponds to the flow from a[i] to b[j]
        num_iterations (int): # of iterations
        eps (float): the coefficient for the entropy regularization
    
    Returns:
        torch.Tensor: an optimal transport plan
    """

    n, m = cost_matrix.shape
    assert a.shape[0] == a.nelement() == n
    assert b.shape[0] == b.nelement() == m
    if a.ndim == 1: a = a[:,None]
    if b.ndim == 1: b = b[:,None]

    k = (-cost_matrix / eps).exp()
    kp = k / a
    u = torch.ones_like(a) / n
    for t in range(num_iterations):
        v = b / (k.T @ u)
        u = 1. / (kp @ v)
        # print(((k.T @ u - b) ** 2).sum())
    s = u * k * v.T

    return s

def sinkhorn_log(
    a: torch.Tensor, b: torch.Tensor, cost_matrix: torch.Tensor,
    num_iterations: int = 50, eps: float = 0.01,
    ab_log: bool = False, returns_log: bool = False
) -> torch.Tensor:
    """ Sinkhorn-Knopp Algorithm

    Args:
        a (torch.Tensor): source distribution
        b (torch.Tensor): drain distribution
        cost_matrix (torch.Tensor): cost matrix; the element cost_matrix[i,j] corresponds to the flow from a[i] to b[j]
        num_iterations (int): # of iterations
        eps (float): the coefficient for the entropy regularization
        ab_log (bool): a and b are given as log-ed value
        returns_log (bool): returned value is log value
    
    Returns:
        torch.Tensor: an optimal transport plan
    """


    n, m = cost_matrix.shape
    assert a.shape[0] == a.nelement() == n
    assert b.shape[0] == b.nelement() == m
    
    loga = a if ab_log else a.log()
    logb = b if ab_log else b.log()
    if a.ndim == 1: a = a[:,None]
    if b.ndim == 1: b = b[:,None]

    logk = -cost_matrix / eps
    logu = torch.zeros_like(loga)
    for t in range(num_iterations):
        logv = logb - (logk + logu).logsumexp(dim=0).unsqueeze(dim=1)
        logu = loga - (logk + logv.T).logsumexp(dim=1).unsqueeze(dim=1)
    logs = logu + logk + logv.T

    return logs if returns_log else logs.exp()
