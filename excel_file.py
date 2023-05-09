import json
import os

import numpy as np
import pandas as pd
from statistics import mean, median

from utils import setup_parser


# Get statistics for each metric and add it to the dataframe.
def dataframe_stats(df, output):
    stats = {}
    for col in df:
        stats[col] = {
            "Min": min(df[col]),
            "Max": max(df[col]),
            "Mean": mean(df[col]),
            "Median": median(df[col]),
            "ST.D": np.std(df[col])
        }

    for st in stats:
        for metric in stats[st]:
            df.loc[metric, st] = stats[st][metric]

    save_dfs(df, output)


# Save the dataframe.
def save_dfs(df, path):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(path, engine="xlsxwriter")
    df.to_excel(writer)
    writer.close()


# Insert the current's stage values and save the dataframe.
def update_dataframe_values(df, patient, evaluation, output):
    for mask in evaluation:
        for stage in evaluation[mask]:
            for metric in evaluation[mask][stage]:
                df.loc[patient, (stage, mask, metric)] = evaluation[mask][stage][metric]

    save_dfs(df, output)

    return df


# Initialise the dataframe with all the necessary rows and columns.
def open_data_frame(patients, evaluation, output):
    masks, stages, metrics = list(evaluation.keys()), [], []
    for mask in evaluation:
        stages.extend(list(evaluation[mask]))
        for metric in evaluation[mask]:
            metrics.extend(list(evaluation[mask][metric]))

    # Remove duplicates while preserving the order.
    stages = list(dict.fromkeys(stages))
    metrics = list(dict.fromkeys(metrics))

    cols = pd.MultiIndex.from_product([stages, masks, metrics])
    df = pd.DataFrame(index=patients, columns=cols)

    save_dfs(df, output)

    return df


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/parser/excel_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    pipeline = args.pl

    df = None
    results_path = f"results/{pipeline}.xlsx"

    exp_path = os.path.join(input_dir, pipeline)
    patients = os.listdir(exp_path)
    for patient in patients:
        patient_dir = os.path.join(exp_path, patient)
        evaluation_path = os.path.join(patient_dir, "evaluation.json")

        if os.path.exists(evaluation_path):
            pf = open(evaluation_path)
            evaluation = json.loads(pf.read())
            pf.close()

            if df is None:
                df = open_data_frame(patients, evaluation, results_path)

            update_dataframe_values(df, patient, evaluation, results_path)

        dataframe_stats(df, results_path)


if __name__ == "__main__":
    main()