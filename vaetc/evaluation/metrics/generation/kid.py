from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
import gc
import torch
import vaetc
from .fid_gen import make_loader

def kid_generation(model: vaetc.models.VAE, dataset: vaetc.data.utils.ImageDataset, batch_size: int = 64):
    """
    returns (mean, biased std) of kid
    """
    
    loader = make_loader(dataset.test_set if isinstance(dataset, vaetc.data.utils.ImageDataset) else dataset, batch_size)

    model.eval()

    kid = KernelInceptionDistance()

    for x, t in tqdm(loader):
        
        x: torch.Tensor

        this_batch_size = x.shape[0]

        kid.update((x.clamp(0., 1.) * 255).to(dtype=torch.uint8).detach(), real=True)
        
        zs = model.sample_prior(this_batch_size)
        xs = model.decode(zs)
        kid.update((xs.clamp(0., 1.) * 255).to(dtype=torch.uint8).detach(), real=False)
    
    kid_mean, kid_std = kid.compute()
    kid_mean = float(kid_mean.detach().cpu().numpy())
    kid_std = float(kid_std.detach().cpu().numpy())
    n_subsets = kid.subsets

    del loader._iterator
    del loader
    del kid
    gc.collect()
    
    return kid_mean, kid_std, n_subsets