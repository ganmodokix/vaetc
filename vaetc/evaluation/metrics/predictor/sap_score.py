import itertools

import numpy as np
import torch

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from vaetc.network.sap_score import score_matrix_continuous

from tqdm import tqdm

def calc_score_matrix(z, t, random_state: int, is_discrete: bool = True):
    """
    returns S \\in [0, 1]^{L \\times T}
    """

    data_size, z_dim = z.shape
    data_size, t_dim = t.shape

    z_train, z_test, t_train, t_test = train_test_split(z, t, test_size=0.2, random_state=random_state)

    if is_discrete:

        score_matrix = np.zeros(shape=(z_dim, t_dim))

        ik = list(itertools.product(range(z_dim), range(t_dim)))
        for i, k in tqdm(ik):


            n_classes = np.unique(t_train[:, k]).size

            if n_classes >= 2:

                # clf = LinearSVC(C=0.01, class_weight="balanced", random_state=random_state) # Segmentation Fault
                clf = LogisticRegression(class_weight="balanced", random_state=random_state)
                clf.fit(z_train[:, i:i+1], t_train[:, k])
                pred = clf.predict(z_test[:, i:i+1])
                pred = np.where(pred > 0.5, 1, 0)
                prob = balanced_accuracy_score(t_test[:, k], pred)
                score_matrix[i,k] = prob
            
            else:

                score_matrix[i,k] = 0.0
        
    else:

        z = torch.tensor(z)
        t = torch.tensor(t)
        score_matrix = score_matrix_continuous(z, t).detach().cpu().numpy()

            
    return score_matrix
            
def sap_score(z: np.ndarray, t: np.ndarray, t_threshold: float = 0.5, random_state=42) -> float:

    # wip
    binary = np.where(t > t_threshold, 1, 0)
    is_discrete = np.mean((t - binary) ** 2) < 0.01

    if is_discrete:
        s = calc_score_matrix(z, binary, random_state=random_state, is_discrete=True)
    else:
        s = calc_score_matrix(z, t, random_state=random_state, is_discrete=False)
    z_dim, t_dim = s.shape

    indices = np.argsort(s, axis=1)
    sap = s[range(z_dim), indices[:,-1]] - s[range(z_dim), indices[:,-2]]

    return float(sap.mean())