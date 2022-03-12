import numpy as np
from scipy.special import logsumexp, erf, xlogy

def sample_gaussian(mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    noise = np.random.standard_normal(size=mean.shape)
    z = mean + np.exp(logvar * 0.5) * noise
    return z

def log_gaussian_density(
    mean1: np.ndarray, logvar1: np.ndarray,
    mean2: np.ndarray, logvar2: np.ndarray
):
    """
    arguments:
    mean1, logvar1: sampling batch outside the log with size (B1, L)
    mean2, logvar2: sampling batch inside the log with size (B2, L)
    returns log(z1|x2) with size (B1, B2, L)
    """

    z1 = sample_gaussian(mean1, logvar1) # (B1, L)

    std2 = np.exp(logvar2 * 0.5) # (B2, L)

    z1      = z1     [:,None,:] # (B1, 1, L)
    mean2   = mean2  [None,:,:] # (B2, L, 1)
    std2    = std2   [None,:,:] # (B2, L, 1)
    logvar2 = logvar2[None,:,:] # (B2, L, 1)
    log2pi  = np.log(np.pi * 2)

    core = (z1 - mean2) / std2 + logvar2 + log2pi

    return -core

def entropy_sampling(mean: np.ndarray, logvar: np.ndarray) -> float:
    """
    arguments:
    mean, logvar: numpy.ndarray with shape (B, L)
    returns float H(z)
    """

    assert mean.ndim == 2
    assert logvar.ndim == 2
    assert mean.shape == logvar.shape

    batch_size = mean.shape[0]
    assert batch_size >= 2

    split_size = batch_size // 2
    mean1   = mean  [:split_size]
    logvar1 = logvar[:split_size]
    mean2   = mean  [split_size:]
    logvar2 = logvar[split_size:]

    size1 = split_size
    size2 = batch_size - split_size
    
    logcore = log_gaussian_density(mean1, logvar1, mean2, logvar2) # (B1, B2, L)
    logcore = np.sum(logcore, axis=2) # (B1, B2)

    entropy = logsumexp(logcore, axis=1) # (B1, )
    entropy = entropy - np.log(size2) # (B1, )
    entropy = -np.mean(entropy) # ()
    entropy = float(entropy)

    return entropy

QUANTIZATION_RANGE = (-4, 4)
QUANTIZATION_BINS = 100
def entropy_quantization(mean: np.ndarray, logvar: np.ndarray, bins: int = QUANTIZATION_BINS):
    """
    arguments:
    mean: np.ndarray of mean of z_i with shape (B, L)
    logvar: np.ndarray of logvar of z_i with shape (B, L)
    returns H(z_i) with shape (L, )
    Note that its sum is NOT the joint entropy of the entire z
    """

    assert bins >= 1
    
    zmin, zmax = QUANTIZATION_RANGE
    poss = np.linspace(zmin, zmax, bins+1)
    a = poss[None,None, :-1]
    b = poss[None,None,1:  ]
    mean   = mean  [:,:,None]
    logvar = logvar[:,:,None]

    std = np.exp(logvar * 0.5)
    sqrt2 = 2 ** 0.5

    ca = (a - mean) / (std * sqrt2)
    cb = (b - mean) / (std * sqrt2)

    # qsin[n,i,j] = Q(s_j|x_n)
    qsin = 0.5 * (erf(cb) - erf(ca)) # (B, L, bins)

    # Q(s_j) = Mean of Q(s_j|x_n) w.r.t. n
    qsi = np.mean(qsin, axis=0) # (L, bins)

    hzi = np.sum(-xlogy(qsi, qsi), axis=1) # (L, )

    return hzi

def entropy_histogram(z: np.ndarray, bins: int = QUANTIZATION_BINS):
    """
    z: shape (B, L)
    returns H(z) via quantization
    """

    assert z.ndim == 2

    batch_size, z_dim = z.shape
    assert z_dim <= 2, "it suffers from curse of dimension"

    eps = 1e-7

    # zmin, zmax = np.min(z, axis=0)[None,:] - eps, np.max(z, axis=0)[None,:] + eps
    # histogram = np.zeros(shape=(bins, ) * z_dim, dtype=int)
    # zpos = ((z - zmin) / (zmax - zmin) * bins).astype(int)
    # zpos = np.clip(zpos, 0, bins-1)
    # for i in range(batch_size):
    #     histogram[tuple(zpos[i])] += 1

    histogram, zposs = np.histogramdd(z, bins=bins)
    zposs = np.stack(zposs)

    histogram = histogram.astype(float)
    prob = histogram / batch_size

    darea = np.prod((zposs[:,-1] - zposs[:,0]) / bins)
    ent = np.sum(-xlogy(prob, prob / darea))

    return ent

def entropy_conditioned(logvar: np.ndarray) -> np.ndarray:
    """
    H(z_i|x)
    """

    return ((logvar + np.log(np.pi * 2) + 1) * 0.5).mean(axis=0)

def entropy_binary(binary: np.ndarray) -> np.ndarray:

    batch_size, t_dim = binary.shape

    p = np.count_nonzero(binary, axis=0) / batch_size

    return -xlogy(p, p) - xlogy(1-p, 1-p)

def entropy_joint_binary(
    mean: np.ndarray, logvar: np.ndarray,
    binary: np.ndarray, threshold: float = 0.5
):
    """
    via quantization.
    returned[i][k] := H(z_i, y_k)
    """

    assert mean.shape == logvar.shape
    assert mean.shape[0] == binary.shape[0]
    
    batch_size, z_dim = mean.shape
    batch_size, t_dim = binary.shape

    mask = (binary >= threshold).T # (t_dim, batch_size)

    ent_joint = []
    for k in range(t_dim):

        n1 = np.count_nonzero(mask[k])
        n0 = batch_size - n1
        
        if min(n0, n1) == 0:

            ent = entropy_quantization(mean, logvar)
            ent_joint += [ent]

        else:

            ent1 = entropy_quantization(mean[mask[k]], logvar[mask[k]])
            ent0 = entropy_quantization(mean[~mask[k]], logvar[~mask[k]])
            ent_joint += [(ent0 * n0 + ent1 * n1) / (n0 + n1)]

    ent_joint = np.stack(ent_joint, axis=1)
    return ent_joint