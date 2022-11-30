from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
import gc
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

        this_batch_size = x.shape[0]

        kid.update(x, real=True)
        
        zs = model.sample_prior(this_batch_size)
        xs = model.decode(zs)
        xs = xs.detach()
        kid.update(xs, real=False)
    
    kid_mean, kid_std = kid.compute()
    kid_mean = float(kid_mean.detach().cpu().numpy())
    kid_std = float(kid_std.detach().cpu().numpy())
    n_subsets = kid.subsets

    del loader._iterator
    del loader
    del kid
    gc.collect()
    
    return kid_mean, kid_std, n_subsets