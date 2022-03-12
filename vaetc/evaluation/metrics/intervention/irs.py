import numpy as np

def irs(z: np.ndarray, t: np.ndarray) -> float:

    data_size, z_dim = z.shape
    data_size, t_dim = t.shape

    if t_dim == 0:
        return float("nan")

    diff_quantile = 1.0

    # [WIP] binarization
    t_threshold = 0.5
    t = np.where(t > t_threshold, 1, 0)

    max_empida = np.max(np.abs(z - z.mean(axis=0, keepdims=True)), axis=0)
    
    factor_empida = np.zeros(shape=(z_dim, t_dim))
    for k in range(t_dim):

        unique_factor_values = np.unique(t[:, k], axis=0)
        num_unique = unique_factor_values.shape[0]

        for i in range(num_unique):

            # E[z_i | y_k]
            mask = t[:,k] == unique_factor_values[i]
            s = z[mask]
            e_loc = np.mean(s, axis=0)

            # EMPIDA(z_i | k, ~k)
            pida = np.abs(s - e_loc)
            mpida = np.percentile(pida, q=diff_quantile*100, axis=0) # 100% percentile === max
            factor_empida[:, k] += mpida

        factor_empida[:, k] /= num_unique

    EPS = 1e-4
    normalized_empida = factor_empida / np.maximum(EPS, max_empida[:, None])
    irs_ik = 1.0 - normalized_empida
    disentanglement_scores = np.max(irs_ik, axis=1)

    if np.sum(max_empida) > 0:
        avg_score = np.average(disentanglement_scores, weights=max_empida)
    else:
        avg_score = np.average(disentanglement_scores)

    return avg_score
