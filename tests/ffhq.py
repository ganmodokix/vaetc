import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "vae",
        "dataset": "ffhq",
        "epochs": 220,
        "batch_size": 256,
        "logger_path": "runs.tests/vae_ffhq_L128",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-3,
            "z_dim": 128,
        }),
        "cuda_sync": True,
        "very_verbose": True,
    })

    print(checkpoint.dataset.test_set[0])

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)