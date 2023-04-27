import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import directed_hausdorff
from statistics import mean, median
import pandas as pd


# Save the dataframe.
def save_dfs(df, path):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(path, engine="xlsxwriter")
    df.to_excel(writer)
    writer.close()


# Insert the current's stage values and save the dataframe.
def update_dataframe_values(df, patient, stage, results, output):
    for mask in results:
        for metric in results[mask]:
            df.loc[patient, (stage, mask, metric)] = results[mask][metric]

    save_dfs(df, output)

    return df


# Initialise the dataframe with all the necessary rows and columns.
def open_data_frame(patients, stages, masks, metrics):
    cols = pd.MultiIndex.from_product([stages, masks, metrics])
    df = pd.DataFrame(index=patients, columns=cols)

    return df


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


# Calculate Dice, Mean Absolute Distance etc. Use comments to include/exclude metrics.
def calculate_metrics(ground_truth, moving):
    ground_truth = nib.load(ground_truth)
    moving = nib.load(moving)

    # For overlap metrics
    ground_truth_data = np.array(ground_truth.get_fdata()).astype(int)
    moving_data = np.array(moving.get_fdata()).astype(int)
    ground_truth_sum = np.sum(ground_truth_data)
    moving_sum = np.sum(moving_data)
    intersection = ground_truth_data & moving_data
    # union = ground_truth_data | moving_data
    intersection_sum = np.count_nonzero(intersection)
    # union_sum = np.count_nonzero(union)

    # For distance metrics
    ground_truth_coords = np.array(np.where(ground_truth_data == 1)).T
    moving_coords = np.array(np.where(moving_data == 1)).T

    # Create the distance matrix with samples because the "on" values on both images are so many that the program
    # overflows the memory while trying to create the distance matrix.
    # n_samples = 10000
    # ground_truth_sample = np.random.choice(ground_truth_coords.shape[0], n_samples, replace=True)
    # moving_sample = np.random.choice(moving_coords.shape[0], n_samples, replace=True)

    # Calculate the distance matrix between the sampled "on" voxels in each image
    # dist_matrix = cdist(ground_truth_coords[ground_truth_sample], moving_coords[moving_sample])

    # Calculate the directed Hausdorff distance between the images
    distance_gdth_2_moving = directed_hausdorff(ground_truth_coords, moving_coords)[0]
    distance_moving_2_gdth = directed_hausdorff(moving_coords, ground_truth_coords)[0]

    return {
        "Dice": 2 * intersection_sum / (ground_truth_sum + moving_sum),
        # "Jaccard": intersection_sum / union_sum,
        # "M.A.D": np.mean(np.abs(dist_matrix)),
        "H.D": max(distance_gdth_2_moving, distance_moving_2_gdth)
    }