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
        
        dfm = dfd.filter(regex="^/metrics/test/", axis=1)

        columns = filter(lambda c: ("Dummy" not in c) and ("(" not in c) and (c != "/metrics/test/JEMMIG"), dfm.columns)
        dfm = dfm.filter(items=columns, axis=1)
        dfm = dfm.sort_index(axis=1)
        column_name_dict = {
            "/metrics/test/DCI Completeness": "DCI-C",
            "/metrics/test/DCI Disentanglement": "DCI-D",
            "/metrics/test/DCI Informativeness": "DCI-I",
            "/metrics/test/Linear Transferability Target Accuracy": "Linear Acc.",
            "/metrics/test/Normalized JEMMIG": "N-JEMMIG",
            "/metrics/test/RMIG": "MIG/RMIG",
            "/metrics/test/WINDIN": "WINDIN"
        }
        dfm.columns = list(map(lambda c: column_name_dict[c], dfm.columns))

        dfm = dfm.dropna(axis=1, how="all")
        dfm.columns = list(map(lambda x: x.split("/")[-1], dfm.columns))

        plt.figure(figsize=(9, 8))
        sns.set()
        
        sns.heatmap(dfm.corr(), vmin=-1, vmax=1, annot=True, cmap="RdYlBu")

        output_path = os.path.join(output_dir_path, f"corr_{dataset}.svg")
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", type=str, help="input path to CSV aggregation")
    parser.add_argument("output_path", type=str, help="output DIRECTORY path")

    args = parser.parse_args()

    main(args.input_path, args.output_path)