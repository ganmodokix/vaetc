import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset

from vaetc.checkpoint import Checkpoint
from vaetc.models import GaussianEncoderAutoEncoderRLModel

def render(checkpoint: Checkpoint):
    
    test_set = checkpoint.dataset.test_set

    n = min(2048, len(test_set))
    loader_test = DataLoader(
        dataset=Subset(test_set, range(n)),
        batch_size=32,
        shuffle=False,
        num_workers=os.cpu_count() - 1,
        pin_memory=True)

    if isinstance(checkpoint.model, GaussianEncoderAutoEncoderRLModel):

        means = []
        logvars = []
        ts = []

        with torch.no_grad():

            for x, t in loader_test:
                
                x = x.to(checkpoint.model.device)
                mean, logvar = checkpoint.model.encode_gauss(x)
                mean = mean.detach().cpu().numpy()
                logvar = logvar.detach().cpu().numpy()
                means.append(mean)
                logvars.append(logvar)

                t = t.detach().cpu().numpy()
                ts.append(t)

            return np.concatenate(means, axis=0), np.concatenate(logvars, axis=0), np.concatenate(ts, axis=0)

    else:

        zs = []
        ts = []

        with torch.no_grad():

            for x, t in loader_test:
                
                x = x.to(checkpoint.model.device)
                z = checkpoint.model.encode(x)
                z = z.detach().cpu().numpy()
                zs.append(z)

                t = t.detach().cpu().numpy()
                ts.append(t)

            return np.concatenate(zs, axis=0), np.concatenate(ts, axis=0)

def plt_scatter_figure(i_broad: int, j_broad: int, zlim):

    plt.figure(figsize=(6, 6))
    sns.set(style="whitegrid")
    plt.xlabel(f"z_{i_broad}")
    plt.ylabel(f"z_{j_broad}")
    plt.xlim(zlim)
    plt.ylim(zlim)

def scatter_plot_gaussian(
    mean: np.ndarray, logvar: np.ndarray,
    i_broad: int, j_broad: int,
    logger_path: str, zlim
):

    num_data, z_dim = mean.shape
    
    plt_scatter_figure(i_broad, j_broad, zlim)

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.set_xlabel(f"z_{i_broad}")
    ax.set_ylabel(f"z_{j_broad}")
    ax.set_xlim(zlim)
    ax.set_ylim(zlim)
    std_i_broad = np.exp(logvar[:,i_broad] * 0.5)
    std_j_broad = np.exp(logvar[:,j_broad] * 0.5)
    
    for idx in range(num_data):
        center = (mean[idx,i_broad], mean[idx,j_broad])
        width, height = std_i_broad[idx] * 2, std_j_broad[idx] * 2
        patch = patches.Ellipse(center, width=width, height=height, fill=False)
        patch.set_edgecolor("k")
        ax.add_patch(patch)
    
    plt.savefig(os.path.join(logger_path, "scatter.svg"))
    plt.savefig(os.path.join(logger_path, "scatter.pdf"))
    plt.close()

def scatter_plot(z: np.ndarray, i_broad: int, j_broad: int, logger_path: str, zlim):
    
    num_data, z_dim = z.shape
    
    plt_scatter_figure(i_broad, j_broad, zlim)

    sns.scatterplot(z[:,i_broad], z[:,j_broad])
    plt.savefig(os.path.join(logger_path, "scatter.svg"))
    plt.savefig(os.path.join(logger_path, "scatter.pdf"))
    plt.close()

def visualize(checkpoint: Checkpoint, i_scatter=None, j_scatter=None, figsize=(6, 6)):

    logger_path = checkpoint.options["logger_path"]

    rendered = render(checkpoint)
    is_gaussian = len(rendered) == 3
    if is_gaussian:
        mean, logvar, t = rendered
        z = mean + np.exp(logvar * 0.5) * np.random.normal(size=mean.shape)
    else:
        z, t = rendered
    num_data, z_dim = z.shape
    num_data, t_dim = t.shape

    # plot empirical latent distributions
    density_path = os.path.join(logger_path, "density")
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

    # visualize the correlation coefficient matrix
    corr = np.corrcoef(z, rowvar=False)
    labels = [f"$z_{{{i}}}$" for i in range(z_dim)]
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1)
    plt.savefig(os.path.join(logger_path, "corr.svg"))
    plt.savefig(os.path.join(logger_path, "corr.pdf"))
    plt.close()

    # select by I(x,z) = H(z) - H(z|x) if gaussian else variance
    if i_scatter is None or j_scatter is None:
        if is_gaussian:
            def gaussian_entropy(logvar):
                return 0.5 * (np.log(2 * np.pi) + logvar + 1)
            hz = gaussian_entropy(np.log(np.var(mean, axis=0)))
            hzx = np.mean(gaussian_entropy(logvar), axis=0)
            ixz = hz - hzx
            mut_rank = np.argsort(ixz)
            i_broad, j_broad = mut_rank[-1], mut_rank[-2]
        else:
            var_rank = np.argsort(np.var(z, axis=0))
            i_broad, j_broad = var_rank[-1], var_rank[-2]
    else:
        i_broad, j_broad = i_scatter, j_scatter

    mean_i_broad = np.mean(z[:,i_broad])
    mean_j_broad = np.mean(z[:,j_broad])
    std_i_broad = np.std(z[:,i_broad])
    std_j_broad = np.std(z[:,j_broad])
    r = np.max([
        np.abs(mean_i_broad - std_i_broad * 4),
        np.abs(mean_i_broad + std_i_broad * 4),
        np.abs(mean_j_broad - std_j_broad * 4),
        np.abs(mean_j_broad + std_j_broad * 4),
    ])
    r = float(r)
    zlim = [-r, r]

    # visualize scatters of z_0 and z_1
    if is_gaussian:
        scatter_plot_gaussian(mean, logvar, i_broad, j_broad, logger_path, zlim)
    else:
        scatter_plot(z, i_broad, j_broad, logger_path, zlim)

    # visualize KDE of z_0 and z_1
    plt.figure(figsize=figsize)
    sns.kdeplot(x=z[:,i_broad], y=z[:,j_broad], fill=True)
    plt.xlabel(f"z_{i_broad}")
    plt.ylabel(f"z_{j_broad}")
    plt.xlim(zlim)
    plt.ylim(zlim)
    plt.savefig(os.path.join(logger_path, "scatter_kde.svg"))
    plt.savefig(os.path.join(logger_path, "scatter_kde.pdf"))
    plt.close()

    if checkpoint.options["dataset"] == "celeba":

        # visualize scatters of z_0 and z_1 by t_20 (Male)
        # zu = (z - np.mean(z, axis=0, keepdims=True)) / np.std(z, axis=0, keepdims=True)
        # tu = (t[:,20] - np.mean(t[:,20], axis=0)) / np.std(t[:,20], axis=0)
        # sxy = np.mean(zu * tu[:,None], axis=0)
        # sx = np.mean(zu ** 2, axis=0)
        # sy = np.mean(tu ** 2)
        # r = sxy / sx / sy
        # print(f"[distribution.py] {r}")
        # argsorted = np.argsort(np.abs(r))
        # i_broad = argsorted[-1]
        # j_broad = argsorted[-2]

        attributes = list(enumerate("5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split(" ")))

        directory_path = os.path.join(logger_path, "scatter")
        os.makedirs(directory_path, exist_ok=True)

        for attr_index, attr_name in tqdm(attributes):

            mask = t[:,attr_index] >= 0.5

            plt.figure(figsize=figsize)
            sns.kdeplot(x=z[~mask,i_broad], y=z[~mask,j_broad], fill=True, alpha=.5, label=f"{attr_name}=-1")
            sns.kdeplot(x=z[ mask,i_broad], y=z[ mask,j_broad], fill=True, alpha=.5, label=f"{attr_name}=1")
            plt.xlabel(f"$z_{i_broad+1}$")
            plt.ylabel(f"$z_{j_broad+1}$")
            plt.xlim(zlim)
            plt.ylim(zlim)
            plt.legend()
            file_name = f"t_{attr_index:03d}_{attr_name}_kde"
            plt.savefig(os.path.join(directory_path, f"{file_name}.pdf"))
            plt.savefig(os.path.join(directory_path, f"{file_name}.svg"))
            plt.close()

            plt.figure(figsize=figsize)
            sns.scatterplot(x=z[~mask,i_broad], y=z[~mask,j_broad], alpha=.5, label=f"{attr_name}=-1")
            sns.scatterplot(x=z[ mask,i_broad], y=z[ mask,j_broad], alpha=.5, label=f"{attr_name}=1")
            plt.xlabel(f"$z_{i_broad+1}$")
            plt.ylabel(f"$z_{j_broad+1}$")
            plt.xlim(zlim)
            plt.ylim(zlim)
            plt.legend()
            file_name = f"t_{attr_index:03d}_{attr_name}"
            plt.savefig(os.path.join(directory_path, f"{file_name}.pdf"))
            plt.savefig(os.path.join(directory_path, f"{file_name}.svg"))
            plt.close()
