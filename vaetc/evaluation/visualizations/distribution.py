import itertools
import os
import gc
from typing import Iterable, Literal, Optional, Union

import cv2
from tqdm import tqdm

from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm, patches, style
import seaborn as sns

import torch
from torch.utils.data import DataLoader, Subset

from vaetc.checkpoint import Checkpoint
from vaetc.models import GaussianEncoderAutoEncoderRLModel
from vaetc.evaluation.preprocess import EncodedData
from vaetc.models.abstract import AutoEncoderRLModel
from vaetc.utils.debug import debug_print

class ScatterFigure:

    def __init__(self, out_name: str) -> None:
        """ out_name: *without* ext """

        self.out_name = out_name

        self.sns_style = "whitegrid"
        self.sns_context = "notebook"
        self.sns_palette = "bright"

        self.figure: Optional[Figure] = None

    def __enter__(self):
        
        self.figure = plt.figure(figsize=(6, 6))
        
        sns.set_theme(context=self.sns_context, style=self.sns_style, palette=self.sns_palette)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        
        self.figure.savefig(self.out_name + ".svg")
        self.figure.savefig(self.out_name + ".pdf")

        plt.close(self.figure)
        self.figure = None

    def ax(self, nrows: int, ncols: int, index: Union[int, tuple[int, int]]) -> Axes:

        return self.figure.add_subplot(nrows, ncols, index)

def scatter_gaussian(ax: Axes,
    meanx: np.ndarray, logvarx: np.ndarray,
    meany: np.ndarray, logvary: np.ndarray,
) -> Axes:

    num_data = len(meanx)
    assert num_data == len(logvarx) == len(meany) == len(logvary)

    stdx = np.exp(logvarx * 0.5)
    stdy = np.exp(logvary * 0.5)

    for idx in range(num_data):
        center = (meanx[idx], meany[idx])
        width, height = stdx[idx] * 2, stdy[idx] * 2
        patch = patches.Ellipse(center, width=width, height=height, fill=False, alpha=0.3)
        patch.set_edgecolor("k")
        ax.add_patch(patch)

    return ax

def scatter(
    data: EncodedData,
    indices: Iterable[int],
    i: int, j: int,
    out_name: str,
    kde: bool = False,
    target_k: Union[int,Literal["categorical"],None] = None,
):

    with ScatterFigure(out_name) as sf:

        ax = sf.ax(1, 1, 1)
        ax.set_xlabel(f"$z_{{{i}}}$")
        ax.set_ylabel(f"$z_{{{j}}}$")

        zi = data.z[indices,i]
        zj = data.z[indices,j]

        xmin = np.min(zi)
        xmax = np.max(zi)
        ymin = np.min(zj)
        ymax = np.max(zj)

        if kde:
            sns.kdeplot(x=zi, y=zj, fill=True, ax=ax)
        else:
            if target_k is not None:
                if target_k == "categorical":
                    path_collection = ax.scatter(x=zi, y=zj, c=np.argmax(data.t[indices], axis=1), cmap="rainbow")
                else:
                    c = data.t[indices,target_k]
                    path_collection = ax.scatter(x=zi, y=zj, c=c, cmap="coolwarm")
            else:
                if data.mean is not None:
                    meani   = data.mean  [indices,i]
                    logvari = data.logvar[indices,i]
                    meanj   = data.mean  [indices,j]
                    logvarj = data.logvar[indices,j]
                    scatter_gaussian(ax, meani, logvari, meanj, logvarj)
                    xmin = np.min(meani - np.exp(logvari * 0.5) * 3)
                    xmax = np.max(meani + np.exp(logvari * 0.5) * 3)
                    ymin = np.min(meanj - np.exp(logvarj * 0.5) * 3)
                    ymax = np.max(meanj + np.exp(logvarj * 0.5) * 3)
                else:
                    path_collection = ax.scatter(x=zi, y=zj)

        # 5% of margin
        margin_ratio = 0.05
        margin_x = max(0.01, xmax - xmin) * margin_ratio
        margin_y = max(0.01, ymax - ymin) * margin_ratio
        xmin, xmax = xmin - margin_x, xmax + margin_x
        ymin, ymax = ymin - margin_y, ymax + margin_y
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        if target_k is not None:
            sf.figure.set_size_inches(7, 6)
            sf.figure.colorbar(path_collection, ax=ax, ticks=np.linspace(0., 1., 6))

def scatter_all(
    data: EncodedData,
    out_dir: str,
    kde=False,
    target_colormap=False,
    z_interest="all",
    t_interest="all",
):

    num_data = data.num_data()
    num_points = 1024
    indices = np.random.choice(num_data, num_points)

    if isinstance(z_interest, str) and z_interest == "all":
        z_interest = list(range(data.z_dim()))
    if isinstance(t_interest, str) and t_interest == "all":
        t_interest = list(range(data.t_dim()))

    os.makedirs(out_dir, exist_ok=True)

    debug_print(f"Scattering {num_points} points out of {num_data} test data ...")
    ij = [(i, j) for i, j in itertools.product(z_interest, z_interest) if i < j]
    ijk = [(i, j, k)
        for i, j in ij
        for k in (t_interest if target_colormap and not isinstance(t_interest, str) else [0])]
    
    for i, j, k in tqdm(ijk):

        if target_colormap and not isinstance(t_interest, str):
            # k is from list

            out_name = f"scatter_z{i:03d}_z{j:03d}_t{k:03d}"
            scatter(data, indices, i, j, os.path.join(out_dir, out_name), kde=kde, target_k=k)

        elif target_colormap == "categorical":
            # k is dummy; target is categorical

            out_name = f"scatter_z{i:03d}_z{j:03d}"
            scatter(data, indices, i, j, os.path.join(out_dir, out_name), kde=kde, target_k="categorical")

        else:
            # no target visualization

            out_name = f"scatter_z{i:03d}_z{j:03d}"
            scatter(data, indices, i, j, os.path.join(out_dir, out_name), kde=kde)

def top_k_interesting_latents(data: EncodedData, k: int):

    z = data.mean if data.mean is not None else data.z
    zstd = z.std(axis=0)
    return np.argsort(zstd)[::-1][:k]

def top_k_interesting_targets(data: EncodedData, k: int):

    zstd = data.t.std(axis=0)
    return np.argsort(zstd)[::-1][:k]

def probably_binary(t: np.ndarray, eps=1e-3):

    d = np.max(np.minimum(np.abs(t), np.abs(1. - t)))
    return d < eps

def probably_categorical(data: EncodedData, eps=1e-3):

    if not probably_binary(data.t, eps):
        return False

    if data.t_dim() < 2:
        return False

    d = np.max(np.minimum(data.t, 1. - data.t))
    if d >= eps:
        return False
    
    indices = np.argsort(data.t, axis=1)
    best = data.t[np.arange(data.num_data()), indices[:,-1]]
    second = data.t[np.arange(data.num_data()), indices[:,-2]]
    disc = best - second
    return bool(1. - np.min(disc) < eps)

def marginal_plot(z: np.ndarray, density_path: str):

    data_size, z_dim = z.shape
    
    os.makedirs(density_path, exist_ok=True)
    for i in tqdm(range(z_dim)):

        data = z[:,i]
        p_skew = stats.skewtest(data).pvalue
        p_kurt = stats.kurtosistest(data).pvalue

        plt.figure()
        sns.histplot(data)
        plt.title(f"z_{i} (skew p={p_skew:.6f}, kurtosis p={p_kurt:.6f})")
        plt.savefig(os.path.join(density_path, f"z_{i:03d}.svg"))
        plt.savefig(os.path.join(density_path, f"z_{i:03d}.pdf"))
        plt.close()

def correlation_plot(z: np.ndarray, out_name: str):

    data_size, z_dim = z.shape

    corr = np.corrcoef(z, rowvar=False)
    labels = [f"$z_{{{i}}}$" for i in range(z_dim)]

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1)
    plt.savefig(out_name + ".svg")
    plt.savefig(out_name + ".pdf")
    plt.close()

@torch.no_grad()
def scatter_decoder(model: AutoEncoderRLModel, z_dim: int, i: int, j: int, out_dir: str):

    os.makedirs(out_dir, exist_ok=True)

    rows, cols = 11, 11

    z = np.zeros(shape=[rows, cols, z_dim])
    pos_y = np.linspace(-3, 3, rows)
    pos_x = np.linspace(-3, 3, cols)
    z[:,:,i] = pos_x[None,:]
    z[:,:,j] = pos_y[:,None]
    z = torch.tensor(z).float().cuda().view(rows * cols, z_dim)

    x2 = model.decode(z)
    x2 = x2.detach().cpu().view(rows, cols, *x2.shape[1:]).numpy()
    img = np.transpose(x2, [0, 1, 3, 4, 2])[...,::-1]
    img = np.concatenate(img, axis=1)
    img = np.concatenate(img, axis=1)
    img = (img * 255).astype(np.uint8)
    
    out_path = os.path.join(out_dir, f"scatter_decoder_z{i:03d}_z{j:03d}.png")
    cv2.imwrite(out_path, img)

def visualize(checkpoint: Checkpoint):

    data = EncodedData(checkpoint.model, checkpoint.dataset.test_set, batch_size=checkpoint.options["batch_size"])

    debug_print("Plotting the latent marginal ...")
    marginal_plot(data.z, os.path.join(checkpoint.options["logger_path"], "density"))

    debug_print("Plotting the latent correlations ...")
    correlation_plot(data.z, os.path.join(checkpoint.options["logger_path"], "corr"))

    z_interest = top_k_interesting_latents(data, 4)
    if probably_categorical(data):
        t_interest = "categorical"
    else:
        t_interest = top_k_interesting_targets(data, 20)

    scatter_all(
        data, os.path.join(checkpoint.options["logger_path"], "scatter"),
        z_interest=z_interest, t_interest=t_interest)
    scatter_all(
        data, os.path.join(checkpoint.options["logger_path"], "scatter_kde"),
        kde=True,
        z_interest=z_interest, t_interest=t_interest)
    scatter_all(
        data, os.path.join(checkpoint.options["logger_path"], "scatter_colormap"),
        target_colormap=True,
        z_interest=z_interest, t_interest=t_interest)

    if isinstance(checkpoint.model, AutoEncoderRLModel):
        debug_print("Scattering the decoder output ...")
        ij = [(i, j) for i, j in itertools.product(z_interest, z_interest) if i < j]
        out_dir = os.path.join(checkpoint.options["logger_path"], "scatter_decoder")
        for i, j in tqdm(ij):
            scatter_decoder(
                checkpoint.model, data.z_dim(),
                i, j,
                out_dir=out_dir)

    del data
    gc.collect()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
