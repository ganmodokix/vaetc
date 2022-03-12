import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "sintrovae",
        "dataset": "ffhq",
        "epochs": 300,
        "batch_size": 256,
        "logger_path": "runs.tests/introvae_ffhq",
        "hyperparameters": yaml.safe_dump({
            "lr": 2e-4,
            "alpha": 0.25,
            "beta": 0.05,
            "margin": 100,
            "z_dim": 16,
        }),
        "cuda_sync": True,
        "very_verbose": True,
        # "until_convergence": True,
        # "patience": 10,
    })

    print(checkpoint.dataset.test_set[0])

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)