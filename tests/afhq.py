import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "dfcvae",
        "dataset": "afhq_v2_cat",
        "epochs": 10,
        "batch_size": 256,
        "logger_path": "runs.tests/afhq_v2_cat_dfcvae",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-4,
            "z_dim": 16,
            "beta": 10,
        }),
        "cuda_sync": False,
        "very_verbose": True,
        "until_convergence": True,
        "patience": 10,
    })

    print(checkpoint.dataset.test_set[0])

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)