import random

import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from vaetc.utils import debug_print


def make_pairs(s: np.array, l: int):

    n = s.shape[0]

    indices0 = np.random.randint(0, n, size=(l, ))
    indices1 = np.random.randint(0, n-1, size=(l, ))
    indices1[indices0 <= indices1] += 1

    return np.abs(s[indices0] - s[indices1])

def make_dataset(z: np.ndarray, t: np.ndarray, size: int = 5000, t_threshold: float = 0.5) -> np.ndarray:

    data_size, z_dim = z.shape
    data_size, t_dim = t.shape

    debug_print("Creating dataset...")

    zdiff = np.empty(shape=(size, z_dim), dtype=z.dtype)
    y = np.empty(shape=(size, ), dtype=np.int)

    available_factors = []
    binary = t > t_threshold
    for k in range(t_dim):
      
        mask = binary[:,k]
        n1 = np.count_nonzero(mask)
        n0 = data_size - n1

        if min(n1, n0) >= 2:
            available_factors.append(k)
    
    for i in tqdm(range(size)):

        y[i] = random.choice(available_factors)

        mask = binary[:,y[i]]
        s0 = z[~mask]
        s1 = z[mask]
        zdiff_i = np.concatenate([make_pairs(s0, l=5), make_pairs(s1, l=5)], axis=0)
        zdiff[i] = np.mean(zdiff_i, axis=0)

    return zdiff, y

def betavae_metric(z: np.ndarray, t: np.ndarray, random_state=42) -> float:

    t_dim = t.shape[1]
    if t_dim == 0:
        return float("nan")

    zdiff, y = make_dataset(z, t)

    zdiff_train, zdiff_test, y_train, y_test = train_test_split(zdiff, y, test_size=0.2, random_state=random_state)

    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(zdiff_train, y_train)

    y_pred = clf.predict(zdiff_test)
    acc = np.count_nonzero(y_pred == y_test) / y_test.shape[0]
    return float(acc)