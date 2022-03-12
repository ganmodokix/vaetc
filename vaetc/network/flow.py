import torch
from torch import nn
from torch.nn import functional as F

class PlanarFlow(nn.Module):
    """ Planar Flow module (https://arxiv.org/abs/1505.05770)
    """

    def __init__(self, dims: int, s: float = 0.01):

        super().__init__()
        
        self.dims = int(dims)
        assert dims > 0, f"# of Dimensions {self.dims} invalid"

        self.u = nn.Parameter(torch.randn(self.dims) * s, requires_grad=True)
        self.w = nn.Parameter(torch.randn(self.dims) * s, requires_grad=True)
        self.b = nn.Parameter(torch.randn(1) * s, requires_grad=True)
        self.enforce_invertibility()
    
    def activation_func(self, affine):
        return torch.tanh(affine)

    def activation_deriv(self, affine):
        return torch.cosh(affine) ** -2

    def enforce_invertibility(self):

        wu = (self.u * self.w).sum()

        eps = 1e-10
        if wu < -1 + eps:

            wn = self.w / torch.norm(self.w) ** 2
            mwu = -1 + F.softplus(wu)

            un = self.u + (mwu - wu) * wn

            self.u.data = un

    def forward(self, *args):

        if len(args) == 1:
            z = args[0]
            logdetj = None
            if isinstance(z, tuple):
                z, logdetj = z
        elif len(args) == 2:
            z, logdetj = args

        self.enforce_invertibility()

        wu = (self.u * self.w).sum()
        affine = (z * self.w[None,:] + self.b[None,:]).sum(dim=1)

        z = z + self.u[None,:] * self.activation_func(affine)[:,None]

        detjm1 = wu * self.activation_deriv(affine)
        logdetj_new = torch.log1p(detjm1)
        if logdetj is not None:
            logdetj = logdetj + logdetj_new
        else:
            logdetj = logdetj_new
        
        return z, logdetj

class LocalPlanarFlow(nn.Module):

    def __init__(self, dims: int, s: float = 0.01):

        super().__init__()
        
        self.dims = int(dims)
        assert dims > 0, f"# of Dimensions {self.dims} invalid"

        self.u = nn.Parameter(torch.randn(self.dims) * s, requires_grad=True)
        self.w = nn.Parameter(torch.randn(self.dims) * s, requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.dims) * s, requires_grad=True)
        self.enforce_invertibility()
    
    def activation_func(self, affine):
        return F.softplus(affine)
        # return torch.tanh(affine)

    def activation_deriv(self, affine):
        return 1 / (1 + torch.exp(-affine))
        # return torch.cosh(affine) ** -2

    def enforce_invertibility(self):

        w = self.w.detach()
        u = self.u.detach()

        wu = u * w

        eps = 1e-10
        mask = wu < -1 + eps

        if not torch.all(mask):

            wn = w[mask] / torch.norm(w[mask]) ** 2
            mwu = -1 + F.softplus(wu[mask])

            un = u[mask] + (mwu - wu[mask]) * wn

            self.u.data[mask] = un

    def forward(self, *args):

        if len(args) == 1:
            z = args[0]
            logdetj = None
            if isinstance(z, tuple):
                z, logdetj = z
        elif len(args) == 2:
            z, logdetj = args

        self.enforce_invertibility()

        wu = self.u * self.w
        affine = z * self.w[None,:] + self.b[None,:]

        z = z + self.u[None,:] * self.activation_func(affine)

        detjm1 = wu[None,:] * self.activation_deriv(affine)
        logdetj_new = torch.log1p(detjm1).sum(dim=1)
        if logdetj is not None:
            logdetj = logdetj + logdetj_new
        else:
            logdetj = logdetj_new
        
        return z, logdetj

class BNFlow(nn.Module):

    def __init__(self, dims: int):

        super().__init__()

        self.dims = int(dims)
        assert self.dims > 0, f"invalid feature num {self.dims}"

        self.bn = nn.BatchNorm1d(self.dims, affine=False, track_running_stats=True)

    def forward(self, args):

        z, logdetj = args

        if self.training:
            var = z.var(dim=0)
        else:
            var = self.bn.running_var.detach()

        z = self.bn(z[:,:,None]).squeeze(dim=2)
        logdetj = logdetj - 0.5 * torch.log(var).sum()

        return z, logdetj