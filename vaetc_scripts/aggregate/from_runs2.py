import os, sys
import argparse
from typing import List
import pandas as pd
import numpy as np
from scipy.stats import sem
from scipy.stats.morestats import Mean

from from_runs import flatten_dict, load

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

MODEL_SELECTION_METRIC = {

    "jemmigvae": "JEMMIG",
    "migsupvae": "MIG-sup",
    "modularityvae": "Modularity Score",
    "njemmigvae": "Normalized JEMMIG",
    "migvae": "RMIG",
    "sapvae": "SAP",
    "dcimigvae": "DCIMIG",
    "windinvae": "WINDIN",
    
    "bvae": "Beta-VAE Metric",
    "btcvae": "RMIG",
    "factorvae": "FactorVAE Metric",
    "infovae": "RMIG",

}

VALIDATED_NOSHOW = [
    "jemmigvae", "migsupvae", "modularityvae", "njemmigvae",
    "migvae", "sapvae", "dcimigvae", "windinvae",
]

MODEL_DISPLAY_NAME = {

    "jemmigvae": "JEMMIG-VAE",
    "migsupvae": "MIG-sup-VAE",
    "modularityvae": "Modularity-VAE",
    "njemmigvae": "NJEMMIG-VAE",
    "migvae": "MIG-VAE",
    "sapvae": "SAP-VAE",
    "dcimigvae": "DCIMIG-VAE",
    "windinvae": "WINDIN-VAE",
    
    "bvae": "$\\beta$-VAE",
    "btcvae": "$\\beta$-TCVAE",
    "factorvae": "FactorVAE",
    "infovae": "InfoVAE",
}

def load_from(runs_dirs: List[str]) -> pd.DataFrame:

    results = load(runs_dirs=args.runs_dirs, mean=False)
    
    fl = [flatten_dict(inst) for inst in results]

    data = []
    for inst in fl:
        inst2 = {}
        for key, value in inst.items():
            if key.startswith("/metrics/"):
                    if key.split("/")[-1] in METRICS:
                        inst2[f"{key}/mean"] = np.mean(value)
                        inst2[f"{key}/sem"] = sem(value)
            else:
                inst2[key] = value
        data += [inst2]

    df = pd.DataFrame(data)
    return df

def table_to_csv(t: List[List[str]]) -> str:
    result = ""
    for row in t:
        result += ",".join(t) + "\\n"
    return result

def table_to_tex(t: List[List[str]]) -> str:
    result = ""
    for i, row in enumerate(t):
        if i == 0: result += "\\toprule\n"
        result += "&".join(row) + "\\\\\n"
        if i == 0: result += "\\midrule\n"
    result += "\\bottomrule\n"
    return result

def hyperparameter_tuning(df: pd.DataFrame) -> pd.DataFrame:

    selected_indices = []

    for (model_name, dataset), dfm in df.groupby(by=["/options/model_name", "/options/dataset"]):
        
        validation_metric = MODEL_SELECTION_METRIC[model_name]
        selection_path = f"/metrics/valid/{validation_metric}/mean"
        criteria = dfm[selection_path]

        if model_name == "jemmigvae":
            selected_index = criteria.idxmin()
        else:
            selected_index = criteria.idxmax()
        
        selected_indices += [selected_index]
        selected = dfm.loc[selected_index]
        print(model_name, dataset, selected_index, selected["/metrics/test/RMIG/mean"])


    dfs = df.loc[selected_indices]

    return dfs

def display_data(dfs: pd.DataFrame) -> pd.DataFrame:

    datan = []
    for index in dfs.index:

        data = {}
        
        data["Model"] = dfs.loc[index, "/options/model_name"]
        data["Dataset"] = dfs.loc[index, "/options/dataset"]
        data["Beta"] = dfs.loc[index, "/hyperparameters/beta"]
        data["Gamma"] = dfs.loc[index, "/hyperparameters/gamma"]
        data["Lambda"] = dfs.loc[index, "/hyperparameters/lambda"]
        data["Alpha"] = dfs.loc[index, "/hyperparameters/alpha"]

        for metric in METRICS.keys():

            if data["Model"] not in VALIDATED_NOSHOW or metric != MODEL_SELECTION_METRIC[data["Model"]]:
            
                metric_mean = dfs.loc[index, f"/metrics/test/{metric}/mean"]
                metric_sem = dfs.loc[index, f"/metrics/test/{metric}/sem"]
                data[f"Metric {metric}"] = f"{metric_mean:.3f}$\\pm${metric_sem * 1.96:.3f}"
                data[f"MeanMetric {metric}"] = metric_mean

            else:

                # metric_mean = dfs.loc[index, f"/metrics/valid/{metric}/mean"]
                # metric_mean = f"{metric_mean:.4g}"
                # if "e" in metric_mean:
                #     a, b = metric_mean.split("e")
                #     if b[0] == "+": b = b[1:]
                #     metric_mean = f"{a}$\\times 10^{{{b}}}$"
                # data[f"Metric {metric}"] = f"{metric_mean}\\dagger"
                data[f"Metric {metric}"] = "--"
                data[f"MeanMetric {metric}"] = -1e100 if metric != "JEMMIG" else 1e100
        
        datan += [data]

    dfn = pd.DataFrame(datan)

    return dfn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dirs", "-r", nargs="*", type=str, default="runs", help="path of runs")
    parser.add_argument("--output_dir", "-o", type=str, default="agg2", help="path to output into")
    args = parser.parse_args()

    df = load_from(args.runs_dirs)

    # ハイパラ調節されたもの
    dfs = hyperparameter_tuning(df)

    # 実験結果として必要な部分だけ
    dfn = display_data(dfs)

    # 表を出力
    model_names = df["/options/model_name"].unique()
    datasets = df["/options/dataset"].unique()

    output_dir = args.output_dir

    print(f"Output into {output_dir}...", file=sys.stderr)
    os.makedirs(output_dir, exist_ok=True)
    dfn.to_csv(os.path.join(output_dir, "from_runs2.csv"))

    for dataset in map(str, datasets):

        dfnd = dfn[dfn["Dataset"] == dataset]

        # 順位
        dfnd_rank = dfnd.copy()
        for column in dfnd_rank.columns:
            if column.startswith("Metric"):
                if column == "Metric JEMMIG":
                    dfnd_rank[column] = dfnd_rank["Mean" + column].rank(ascending=True)
                else:
                    dfnd_rank[column] = dfnd_rank["Mean" + column].rank(ascending=False)
        
        # 表示用の列名
        def column_map(x: str):
            if x.startswith("Metric"):
                return METRICS[x.replace("Metric ", "")]
            elif x == "Beta":
                return "$\\beta$"
            elif x == "Gamma":
                return "$\\gamma$"
            elif x == "Alpha":
                return "$\\alpha$"
            elif x == "Lambda":
                return "$\\lambda$"
            else:
                return x

        # 表示用のテーブル
        table = []

        columns = dfnd.columns
        columns = filter(lambda x: "MeanMetric" not in x, columns)
        columns = map(column_map, columns)
        table.append(list(columns))

        # 表を作成
        for i in dfnd.index:

            line = []
            
            for j in dfnd.columns:
                
                if j.startswith("MeanMetric"): continue

                if j.startswith("Metric"):
                    first = 0.5 <= dfnd_rank.loc[i,j] <= 1.5
                    second = 1.5 <= dfnd_rank.loc[i,j] <= 2.5
                else:
                    first = False
                    second = False

                term = str(dfnd.loc[i,j])
                if term == "nan":
                    term = "--"
                elif first:
                    # rank = dfnd_rank[j]
                    # mean_first = dfnd.loc[(0.5 <= rank) & (rank <= 1.5),j]
                    # mean_second = dfnd.loc[(1.5 <= rank) & (rank <= 2.5),j]
                    # print(mean_first.to_dict().values().__iter__().__next__())
                    # print(mean_second.to_dict().values().__iter__().__next__())
                    term = f"\\metricfirst{{{term}}}"
                elif second:
                    term = f"\\metricsecond{{{term}}}"

                line += [term]

            # fp.write(" & ".join(line) + "\\\\\n")
            table.append(line)

        models_raw = list(MODEL_SELECTION_METRIC.keys())
        table = [table[0], *sorted(table[1:], key=lambda row: models_raw.index(row[0]))]

        table = list(zip(*table)) # transpose
        table[0] = list(map(lambda x: MODEL_DISPLAY_NAME[x] if x in MODEL_DISPLAY_NAME else x, table[0]))
        
        with open(os.path.join(output_dir, f"from_runs2_{dataset}.txt"), "w") as fp:
            fp.write(table_to_tex(table))