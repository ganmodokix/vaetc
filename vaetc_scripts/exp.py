import sys
import os

import itertools
import datetime
import subprocess
import argparse

import numpy as np
import torch

import yaml

import vaetc

def experiment(
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    logger_path: str,
    hyperparameters: dict,
):

    checkpoint = vaetc.Checkpoint(options={
        "model_name": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "logger_path": logger_path,
        "hyperparameters": hyperparameters,
    })

    vaetc.fit(checkpoint)
    # vaetc.evaluate(checkpoint)

def continue_experiment(
    checkpoint_path: str,
    python: str
):

    subprocess.run([python, "continue.py", "--checkpoint_path", checkpoint_path])

def product_dict(d: dict):

    keys = list(d.keys())
    values_comb = itertools.product(*d.values())

    results = []
    for values in values_comb:
        results += [dict(zip(keys, values))]
    
    return results

def hyperparameter_id(d: dict):

    keys = list(d.keys())
    keys.sort()

    return "-".join(map(lambda key: f"""{key}{d[key]}""", keys))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", "-r", type=str, default="runs", help="output dir")
    parser.add_argument("--dry_run", "-d", action="store_true", help="dry run")
    args = parser.parse_args()

    exps = [
        {
            "dataset": ["celeba"],
            "model_name": ["infovae"],
            "hyperparameters": {
                "lr": [1e-4],
                "z_dim": [64],
                "alpha": [0],
                "lambda": [0.01, 0.1, 1, 10, 100],
            },
        },
        {
            "dataset": ["mnist", "stl10"],
            "model_name": ["infovae"],
            "hyperparameters": {
                "lr": [1e-4],
                "z_dim": [16],
                "alpha": [0],
                "lambda": [0.01, 0.1, 1, 10, 100],
            },
        },

    ]
    
    epochs = 50
    batch_size = 256

    for settings in exps:

        for dataset in settings["dataset"]:

            for model_name in settings["model_name"]:

                for hyperparameters in product_dict(settings["hyperparameters"]):

                    now = datetime.datetime.now()
                    hid = hyperparameter_id(hyperparameters)
                    id = f"{dataset}_{model_name}_{hid}"
                    logger_path = f"{args.runs_dir}/{id}"

                    print(f"Experiment going on {logger_path}", file=sys.stderr)
                    
                    if not os.path.exists(logger_path):

                        if not args.dry_run:
                            experiment(
                                model_name=model_name,
                                dataset=dataset,
                                epochs=epochs,
                                batch_size=batch_size,
                                logger_path=logger_path,
                                hyperparameters=hyperparameters,
                                python=args.python)
                    
                    else:

                        to_be_continued = False
                        
                        checkpoint_path = os.path.join(logger_path, "checkpoint_last.pth")
                        if os.path.exists(checkpoint_path):
                            state_dict = torch.load(checkpoint_path)
                            trained_epochs = state_dict["training_state"]["epochs"]
                            total_epochs = state_dict["options"]["epochs"]
                            if trained_epochs < total_epochs:
                                to_be_continued = True

                        if to_be_continued:

                            print(f"Results already exist but trained {trained_epochs}/{total_epochs} epochs; continue", file=sys.stderr)

                            if not args.dry_run:
                                continue_experiment(checkpoint_path, args.python)

                        else:

                            print(f"Results already exist; skipped", file=sys.stderr)
