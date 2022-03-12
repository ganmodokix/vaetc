import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "sintrovae",
        "dataset": "getchu",
        "epochs": 256,
        "batch_size": 256,
        "logger_path": "runs.tests/sintrovae_getchu_res2",
        "hyperparameters": yaml.safe_dump({
            "lr": 2e-4,
            "z_dim": 64,
            "beta_rec": 1,
            "beta_kl": 1,
            "beta_neg": 128,
            "gamma_r": 1e-8,
        }),
        "cuda_sync": False,
        "very_verbose": True,
        # "until_convergence": True,
        # "patience": 10,
    })

    print(checkpoint.dataset.test_set[0])

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)