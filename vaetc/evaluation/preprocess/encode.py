import os
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from vaetc.checkpoint import Checkpoint
from vaetc.models import RLModel, AutoEncoderRLModel, GaussianEncoderAutoEncoderRLModel
from vaetc.utils import debug_print

def encode_set(model: RLModel, dataset: Dataset, batch_size: int):

    xs, ts, zs = [], [], []
    means, logvars = [], []
    x2s = []

    loader = DataLoader(dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count() - 1)

    is_decodable = isinstance(model, AutoEncoderRLModel)
    is_gaussian = isinstance(model, GaussianEncoderAutoEncoderRLModel)

    # encoding batches

    debug_print("Encoding the set...")
    for x, t in tqdm(loader):

        x = x.cuda()
        
        if is_gaussian:
            mean, logvar = model.encode_gauss(x)
            z = mean + (logvar * 0.5).exp() * torch.randn_like(logvar)
        else:
            z = model.encode(x)
            mean, logvar = None, None
        
        if is_decodable:
            x2 = model.decode(z)
        else:
            x2 = None

        x = x.detach().cpu().numpy()
        t = t.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        if is_decodable:
            x2 = x2.detach().cpu().numpy()
        if is_gaussian:
            mean = mean.detach().cpu().numpy()
            logvar = logvar.detach().cpu().numpy()

        xs += [x]
        ts += [t]
        zs += [z]
        means += [mean]
        logvars += [logvar]
        x2s += [x2]

    # concatenate batches

    x = np.concatenate(xs, axis=0)
    t = np.concatenate(ts, axis=0)
    z = np.concatenate(zs, axis=0)

    if is_decodable:
        x2 = np.concatenate(x2s, axis=0)
    else:
        x2 = None

    if is_gaussian:
        mean = np.concatenate(means, axis=0)
        logvar = np.concatenate(logvars, axis=0)
    else:
        mean, logvar = None, None

    return x, t, z, mean, logvar, x2, xs, ts, zs, means, logvars, x2s

class EncodedData:

    batch_size: int
    x: np.ndarray
    t: np.ndarray
    z: np.ndarray
    mean: Optional[np.ndarray]
    logvar: Optional[np.ndarray]
    xs: list[np.ndarray]
    ts: list[np.ndarray]
    zs: list[np.ndarray]
    means: list[Optional[np.ndarray]]
    logvars: list[Optional[np.ndarray]]

    def __init__(self, model: RLModel, dataset: Dataset, batch_size: int = 32) -> None:
        
        self.batch_size = int(batch_size)

        x, t, z, mean, logvar, x2, xs, ts, zs, means, logvars, x2s = encode_set(model, dataset, batch_size)
        assert x.shape[0] == t.shape[0] == z.shape[0]
        assert (mean is None) or (z.shape == mean.shape == logvar.shape)

        # raw
        self.x = x
        self.t = t
        self.z = z
        self.mean = mean
        self.logvar = logvar
        self.x2 = x2

        # batches
        self.xs = xs
        self.ts = ts
        self.zs = zs
        self.means = means
        self.logvars = logvars
        self.x2s = x2s

    def iter_batch(self):
        return zip(self.xs, self.ts, self.zs, self.means, self.logvars, self.x2s)

    def num_batch(self):
        return len(self.xs)

    def is_gaussian(self):
        return self.mean is not None

    def z_dim(self):
        return self.z.shape[1]

    def t_dim(self):
        return self.t.shape[1]