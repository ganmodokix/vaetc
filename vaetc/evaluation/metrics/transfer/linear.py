from typing import Tuple
from tqdm import tqdm
import numpy as np

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_validate

def linear_transferability(z: np.ndarray, t: np.ndarray, t_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:

    assert z.shape[0] == t.shape[0]
    
    data_size, z_dim = z.shape
    data_size, t_dim = t.shape

    binary = np.where(t >= t_threshold, 1, 0)

    acc = []
    acc_dummy = []
    
    for i in tqdm(range(t_dim)):
        y = binary[:,i]
        n_ones = np.count_nonzero(y)
        if min(n_ones, data_size - n_ones) == 0:
            # obvious case
            acc += [1.0]
        else:
            score = 0
            for c in [0.3, 0.03, 0.003]:
                classifier = LogisticRegression(penalty="l1", solver="liblinear", C=c)
                result = cross_validate(classifier, z, y, scoring="accuracy", cv=5, n_jobs=-1)
                result = np.mean(result["test_score"])
                score = max(score, result)
            acc += [score]
        
        acc_dummy += [max(n_ones, data_size - n_ones) / data_size]
    
    return acc, acc_dummy
