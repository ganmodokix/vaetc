import sys
import itertools
from typing import Tuple

import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from scipy.special import xlogy

def select_rf(z: np.ndarray, t: np.ndarray, random_state: int):

    assert z.ndim == 2
    assert z.shape[0] == t.shape[0]

    batch_size, z_dim = z.shape
    _, t_dim = t.shape

    # t = (t >= t_threshold).astype(np.uint8)
    
    hyperparameters = {
        "max_depth": [2, 4, 8, 12],
        "n_estimators": [10],
        "n_jobs": [-1],
    }
    
    names = list(hyperparameters.keys())
    indices = itertools.product(*hyperparameters.values())

    best_score, best_kwargs = float("inf"), {}

    for index in list(indices):

        kwargs = dict(zip(names, index))
        estimator = RandomForestRegressor(**kwargs, random_state=random_state)
        scores = []
        for i in tqdm(range(t_dim)):
            score_i_cv = cross_validate(estimator, z, t[:,i],
                scoring="neg_mean_squared_error", n_jobs=-1, cv=5)
            score_i = -np.mean(score_i_cv["test_score"])
            scores.append(score_i)
        entire_score = np.mean(scores)

        if best_score > entire_score:
            best_score = entire_score
            best_kwargs = kwargs
        
        print(f"[classifier.py: select_rf()] {entire_score:.6f} in {kwargs}", file=sys.stderr)

    print(f"[classifier.py: select_rf()] {best_score:.6f} in {best_kwargs}", file=sys.stderr)

    return RandomForestRegressor(**best_kwargs)

def disentanglement(gini: np.ndarray) -> float:

    z_dim, t_dim = gini.shape
    
    p = np.abs(gini)
    p = p / np.maximum(1e-12, np.sum(p, axis=1, keepdims=True))

    d = 1 + np.sum(xlogy(p, p) / np.log(t_dim), axis=1)

    rho = np.sum(gini, axis=1) / np.maximum(1e-12, np.sum(gini))

    return float(np.sum(rho * d))

def completeness(gini: np.ndarray) -> np.ndarray:

    z_dim, t_dim = gini.shape
    
    p = np.abs(gini)
    p = p / np.maximum(1e-12, np.sum(p, axis=0, keepdims=True))

    d = 1 + np.sum(xlogy(p, p) / np.log(z_dim), axis=0)

    return d

def explicitness(pred_error: np.ndarray) -> np.ndarray:

    return np.maximum(0, 1 - 6 * pred_error)

def dci_score(z: np.ndarray, t: np.ndarray, random_state: int = 42) -> Tuple[float, np.ndarray, np.ndarray]:

    assert z.ndim == 2
    assert z.shape[0] == t.shape[0]

    batch_size, z_dim = z.shape
    _, t_dim = t.shape

    z_train, z_test, t_train, t_test = train_test_split(z, t, test_size=0.33, random_state=random_state)

    estimator = select_rf(z_train, t_train, random_state)

    gini = np.empty(shape=(z_dim, t_dim), dtype=np.float32)
    pred_error = np.empty(shape=(t_dim, ), dtype=np.float32)

    for i in tqdm(range(t_dim)):

        estimator.fit(z_train, t_train[:,i])

        square_error = (estimator.predict(z_test) - t_test[:,i]) ** 2
        pred_error[i] = np.mean(square_error)
        gini[:,i] = estimator.feature_importances_

    return disentanglement(gini), completeness(gini), explicitness(pred_error)

# unit tests
if __name__ == "__main__":

    print("independent case:")
    t = np.random.uniform(size=(2048, 10))
    z = np.random.normal(size=(2048, 10))
    d, c, i = dci_score(z, t)
    print(d, c.mean(), i.mean())

    print("dependent case:")
    t = np.random.uniform(size=(2048, 10))
    z = t @ np.random.normal(size=(10, 10))
    d, c, i = dci_score(z, t)
    print(d, c.mean(), i.mean())

    print("perfect case:")
    t = np.random.uniform(size=(2048, 10))
    z = t
    d, c, i = dci_score(z, t)
    print(d, c.mean(), i.mean())

    print("lo-mod hi-com:")
    t = np.random.uniform(size=(2048, 10))
    z = np.random.normal(size=(2048, 10)) * 2 - 1
    z[:,:1] = t @ np.random.normal(size=(10, 1))
    d, c, i = dci_score(z, t)
    print(d, c.mean(), i.mean())

    print("hi-mod lo-com:")
    t = np.random.uniform(size=(2048, 3))
    z = np.concatenate([
        t[:,0:1] @ np.random.normal(size=(1, 10)),
        t[:,1:2] @ np.random.normal(size=(1, 10)),
        t[:,2:3] @ np.random.normal(size=(1, 10)),
    ], axis=-1)
    d, c, i = dci_score(z, t)
    print(d, c.mean(), i.mean())