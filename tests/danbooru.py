import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "vaegan",
        "dataset": "danbooru",
        "epochs": 1000,
        "batch_size": 256,
        "logger_path": "runs.tests/vaegan_danbooru4",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-3,
            "lr_disc": 1e-3,
            "beta": 8,
            "gamma": 0.01,
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