import random

import numpy as np
from numpy.core.fromnumeric import argmin, shape

from tqdm import tqdm
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from vaetc.utils import debug_print

def make_dataset(z: np.ndarray, t: np.ndarray, size: int = 5000, t_threshold: float = 0.5) -> np.ndarray:

    data_size, z_dim = z.shape
    data_size, t_dim = t.shape

    debug_print("Creating dataset...")

    # rescaled representation
    EPS = 1e-4
    z = z / np.maximum(EPS, np.std(z, axis=0, keepdims=True))

    argminstd = []
    y = []

    available_factors = []
    binary = t > t_threshold
    for k in range(t_dim):
      
        mask = binary[:,k]
        n1 = np.count_nonzero(mask)
        n0 = data_size - n1

        if min(n1, n0) >= 2:
            available_factors.append(k)
    
    for i in tqdm(range(size)):

        factor = random.choice(available_factors)
        value = random.choice([True, False])

        mask = binary[:,factor] == value
        mask_indices = np.nonzero(mask)[0]
        indices = np.random.choice(mask_indices, size=(100, ))
        s = z[indices]

        argminstd.append(np.argmin(np.var(s, axis=0)))
        y.append(factor)

    return np.array(argminstd), np.array(y)

def factorvae_metric(z: np.ndarray, t: np.ndarray, random_state=42) -> float:

    data_size, z_dim = z.shape
    data_size, t_dim = t.shape

    if t_dim == 0:
        return float("nan")

    argminstd, y = make_dataset(z, t)
    
    argminstd_train, argminstd_test, y_train, y_test = train_test_split(argminstd, y, test_size=0.2, random_state=random_state)
    test_size = argminstd_test.shape[0]
    
    v, *v_ranges = np.histogramdd(np.stack([argminstd_train, y_train], axis=1), bins=(np.arange(z_dim+1)-0.5, np.arange(t_dim+1)-0.5))
    majorities = np.argmax(v, axis=1) # (L, )

    y_pred = np.zeros(shape=(test_size, ), dtype=int)
    for j in range(z_dim):
        y_pred[argminstd_test == j] = majorities[j]

    acc = np.count_nonzero(y_pred == y_test) / test_size
    return float(acc)
