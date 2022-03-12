from typing import Tuple

import numpy as np
import torch

import vaetc.network.doetal as doetal

def np2pt(arr: np.ndarray):
    return torch.tensor(arr).cuda()

def pt2np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

def misjed(mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    return pt2np(doetal.misjed(np2pt(mean), np2pt(logvar)))

def informativeness(mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    return pt2np(doetal.informativeness(np2pt(mean), np2pt(logvar)))

def windin(mean: np.ndarray, logvar: np.ndarray) -> float:
    return float(pt2np(doetal.windin(np2pt(mean), np2pt(logvar))))

def rmig_jemmig(
    mean: np.ndarray, logvar: np.ndarray,
    binary: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rmig_k, jemmig_k, normalized_jemmig_k = doetal.rmig_jemmig(np2pt(mean), np2pt(logvar), np2pt(binary))
    return pt2np(rmig_k), pt2np(jemmig_k), pt2np(normalized_jemmig_k)

def mig_sup(
    mean: np.ndarray, logvar: np.ndarray,
    binary: np.ndarray
) -> np.ndarray:
    return pt2np(doetal.mig_sup(np2pt(mean), np2pt(logvar), np2pt(binary)))

def modularity(
    mean: np.ndarray, logvar: np.ndarray,
    binary: np.ndarray
) -> np.ndarray:
    return pt2np(doetal.modularity(np2pt(mean), np2pt(logvar), np2pt(binary)))

def dcimig(
    mean: np.ndarray, logvar: np.ndarray,
    binary: np.ndarray
) -> float:
    return float(pt2np(doetal.dcimig(np2pt(mean), np2pt(logvar), np2pt(binary))))