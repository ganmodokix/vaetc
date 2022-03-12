import os
import argparse
from typing import Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

from multiprocessing import Pool

def load_entry(entry_path: str, mean=True):

    path_to_options = os.path.join(entry_path, "options.yaml")
    if os.path.isfile(path_to_options):
        with open(path_to_options, "r") as fp:
            options = yaml.safe_load(fp)

    metrics = {}
    for set_name in ["valid", "test"]:

        if mean:
            
            path_to_metrics = \
                os.path.join(entry_path, f"metrics_{set_name}.yaml")
            if os.path.isfile(path_to_metrics):
                with open(path_to_metrics, "r") as fp:
                    metrics[set_name] = yaml.safe_load(fp)
        else:

            sup_k = 0
            while True:
                path_to_metrics = os.path.join(entry_path, f"metrics_{set_name}_{sup_k}.yaml")
                if not os.path.exists(path_to_metrics): break
                sup_k += 1
            
            metrics[set_name] = {}
            for k in range(sup_k):
                path_to_metrics = os.path.join(entry_path, f"metrics_{set_name}_{k}.yaml")
                with open(path_to_metrics, "r") as fp:
                    metrics_k = yaml.safe_load(fp)
                    for key in metrics_k:
                        if key in metrics[set_name]:
                            metrics[set_name][key] += [metrics_k[key]]
                        else:
                            metrics[set_name][key] = [metrics_k[key]]

    inst = {
        "options": options,
        "hyperparameters": yaml.safe_load(options["hyperparameters"]),
        "metrics": metrics,
    }

    return inst

def load_entry_unagg(entry_path: str):
    return load_entry(entry_path, mean=False)

def load(runs_dirs: str = ["runs"], mean=True) -> List[Dict[str, dict]]:
    """Load and aggregate results as a list
    """

    results = []

    dir_list = []
    for runs_dir in runs_dirs:
        with os.scandir(runs_dir) as scd:
            for entry in scd:
                if entry.is_dir():
                    dir_list += [entry]
    
    # for entry in tqdm(dir_list):
        
    #     inst = load_entry(entry_path=entry.path, mean=mean)
    #     results += [inst]

    entry_paths = [entry.path for entry in dir_list]
    with Pool() as p:
        iter = p.imap(load_entry if mean else load_entry_unagg, entry_paths)
        results = list(tqdm(iter, total=len(entry_paths)))

    return results

def flatten_dict(obj, result = None, prefix = ""):

    if result is None:
        result = {}

    for name in obj:
        if isinstance(obj[name], dict):
            flatten_dict(obj[name], result=result, prefix=f"{prefix}/{name}")
        else:
            result[f"{prefix}/{name}"] = obj[name]
    
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dirs", "-r", nargs="*", type=str, default="runs", help="path of runs")
    parser.add_argument("--output_name", "-o", type=str, default="aggregate/aggregated.yaml", help="output path")
    args = parser.parse_args()

    output_basename = ".".join(args.output_name.split(".")[:-1])
    print(output_basename)
    
    results = load(runs_dirs=args.runs_dirs)

    with open(f"{output_basename}.yaml", "w") as fp:
        yaml.safe_dump(results, fp)

    fl = [flatten_dict(inst) for inst in results]
    
    df = pd.DataFrame(fl)

    df.to_csv(f"{output_basename}.csv")