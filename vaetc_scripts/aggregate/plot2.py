import argparse
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

def main(input_csv_path, output_dir_path):

    df = pd.read_csv(input_csv_path)

    os.makedirs(output_dir_path, exist_ok=True)

    datasets = df["/options/dataset"].unique()
    for dataset in datasets:
        
        dfd = df[df["/options/dataset"] == dataset].dropna(axis=1, how="all")
        
        metric_columns = dfd.filter(regex="^/metrics/test/", axis=1).columns.unique()

        for y in metric_columns:
            
            metric_name = y.split("/")[-1]
            
            plt.figure()
            sns.set()
            plt.xscale("log")
            
            sns.scatterplot(data=dfd, x="/hyperparameters/beta", y=y)
            
            plt.xlabel("$\\beta$")
            plt.ylabel(metric_name)
            plt.title(f"Model: $\\beta$-VAE, Dataset: {dataset}")

            output_name = os.path.join(output_dir_path, f"beta-{metric_name}_{dataset}.svg")
            plt.savefig(output_name)
            plt.close()
            print(f"Done at {output_name}", file=sys.stderr)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", type=str, help="input path to CSV aggregation")
    parser.add_argument("output_path", type=str, help="output DIRECTORY path")

    args = parser.parse_args()

    main(args.input_path, args.output_path)