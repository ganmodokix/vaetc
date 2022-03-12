import torch

def score_matrix_continuous(z: torch.Tensor, t: torch.Tensor, eps=1e-4) -> torch.Tensor:

    data_size, z_dim = z.shape
    data_size, t_dim = t.shape
    assert data_size > 1

    cov = torch.mean((z - z.mean(dim=0, keepdim=True))[:,:,None] * (t - t.mean(dim=0, keepdim=True))[:,None,:], dim=0)
    var_z = torch.mean((z - z.mean(dim=0, keepdim=True)) ** 2, dim=0)
    var_t = torch.mean((t - t.mean(dim=0, keepdim=True)) ** 2, dim=0)

    return cov ** 2 / (var_z[:,None] * var_t[None,:] + eps)

def sap_score_continuous(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    returns SAP[i]
    """

    s = score_matrix_continuous(z, t)
    z_dim, t_dim = s.shape

    indices = torch.argsort(s, dim=1)
    sap = s[range(z_dim), indices[:,-1]] - s[range(z_dim), indices[:,-2]]

    return sap