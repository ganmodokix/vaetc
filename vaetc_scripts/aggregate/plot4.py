import argparse
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

X_VARIABLE = "/hyperparameters/beta"
METRICS = {
    "DCI Disentanglement": "DCI-D",
    "DCI Completeness": "DCI-C",
    "DCI Informativeness": "DCI-I",
    "Beta-VAE Metric": "Z-diff",
    "FactorVAE Metric": "Z-min",
    "IRS": "IRS",
    "JEMMIG": "JEMMIG",
    "MIG-sup": "MIG-sup",
    "Modularity Score": "Modularity",
    "Normalized JEMMIG": "NJEMMIG",
    "RMIG": "MIG",
    "SAP": "SAP",
    "DCIMIG": "DCIMIG",
    "WINDIN": "WINDIN",
}

def main(input_dir_path, output_dir_path):

    os.makedirs(output_dir_path, exist_ok=True)

    path_to_cache = os.path.join(output_dir_path, "plot4_cache.yaml")
    if not os.path.exists(path_to_cache):
    
        data = []

        scd = list(os.scandir(input_dir_path))
        for entry in tqdm(scd):

            if not entry.is_dir: continue

            path_to_options = os.path.join(entry.path, "options.yaml")
            path_to_mean = os.path.join(entry.path, "metrics_test.yaml")
            path_to_sem = os.path.join(entry.path, "metrics_test_se.yaml")

            with open(path_to_options, "r") as fp:
                options = yaml.safe_load(fp)

            with open(path_to_mean, "r") as fp:
                mean = yaml.safe_load(fp)

            with open(path_to_sem, "r") as fp:
                sem = yaml.safe_load(fp)

            mean = {name: value for name, value in mean.items() if name in METRICS}
            sem = {name: value for name, value in sem.items() if name in METRICS}

            data.append({"options": options, "mean": mean, "sem": sem})

        with open(path_to_cache, "w", encoding="utf8") as fp:
            yaml.safe_dump(data, fp)

    else:

        with open(path_to_cache, "r", encoding="utf8") as fp:
            data = yaml.safe_load(fp)

    datasets = set(map(lambda x: x["options"]["dataset"], data))

    for dataset in datasets:

        data_subset = [entry for entry in data if entry["options"]["dataset"] == dataset]

        betas = [yaml.safe_load(row["options"]["hyperparameters"])["beta"] for row in data_subset]
        means = {name: [(row["mean"][name] if name in row["mean"] else np.nan) for row in data_subset] for name in METRICS}
        sems = {name: [(row["sem"][name] if name in row["sem"] else np.nan) for row in data_subset] for name in METRICS}

        # sns.set()
        fig = plt.figure(figsize=(12,3))
        plt.subplots_adjust(left=0.05, right=0.87, bottom=0.15, top=0.9)
        bbox_to_anchor_legend = (1.01, 1.14)

        # fig = plt.figure(figsize=(6.4, 4.8))
        # plt.subplots_adjust(left=0.1, right=0.77, bottom=0.1, top=0.99)
        # bbox_to_anchor_legend = (1.01, 1)

        plt.xscale("log")
        plt.xlabel(r"$\beta$")
        plt.ylabel("Normalized Metric Value")
        plt.title({"celeba": "Dataset: CelebA", "mnist": "Dataset: MNIST", "stl10": "Dataset: STL-10"}[dataset])

        linestyles = ["solid", "dotted", "dashed", "dashdot"] * 100

        for i, (metric_path, metric_name) in enumerate(METRICS.items()):
            
            x = np.array(betas)
            ym = np.array(means[metric_path])
            if metric_path == "JEMMIG":
                ym *= -1
            ys = np.array(sems[metric_path])

            idx = np.argsort(x)
            mask = (1 - 1e-10 < x[idx]) & (x[idx] < 1 + 1e-10)
            kab = mask.argmax()
            idx = np.delete(idx, kab)
            
            x = x[idx]
            ym = ym[idx]
            ys = ys[idx]

            ymmin, ymmax = np.min(ym), np.max(ym)
            ymn = (ym - ymmin) / (ymmax - ymmin)
            ysn = ys / (ymmax - ymmin)
            
            # # 95% interval
            # plt.fill_between(x=x, y1=ymn - ysn * 2, y2=ymn + ysn * 2, alpha=0.3)

            # main
            plt.plot(x, ymn, label=metric_name, linestyle=linestyles[i])

        plt.ylim([-0.1, 1.1])
        plt.legend(bbox_to_anchor=bbox_to_anchor_legend)

        path_to_output = os.path.join(output_dir_path, f"plot4_{dataset}")
        plt.savefig(path_to_output + ".pdf")
        plt.savefig(path_to_output + ".svg")
        plt.close()

        print(path_to_output)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", type=str, help="input DIRECTORY path")
    parser.add_argument("output_path", type=str, help="output DIRECTORY path")

    args = parser.parse_args()

    main(args.input_path, args.output_path)