import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist

from shutil import rmtree


# Set up the parser and the correct argument messages loaded from the respective json file.
def setup_parser(json_file):
    with open(json_file, 'r') as messages:
        messages = json.load(messages)

        parser = argparse.ArgumentParser(description=messages["description"])
        required_args = parser.add_argument_group("required arguments")

        for cat in messages["arguments"]:
            for arg in messages["arguments"][cat]:
                message = messages["arguments"][cat][arg]
                required_args.add_argument(arg, help=message, required=True if cat == "required" else False)

        return parser.parse_args()


# Validate paths, create output paths if they do not exist
def validate_paths(input_dir, output_dir):
    # Validate that arguments are not None.
    if (not input_dir) or (not output_dir):
        sys.exit("Specify input -inp, output -out and processed output -pout.")

    # Validate that input directory exists.
    if not os.path.exists(input_dir):
        sys.exit("This input directory does not exist.")


# Create a directory.
def create_dir(output_path, dir_name):
    dir_output_path = os.path.join(output_path, dir_name)
    if not os.path.exists(dir_output_path):
        os.mkdir(dir_output_path)

    return dir_output_path


# Delete the directory if it exists.
def delete_dir(directory):
    if os.path.exists(directory):
        rmtree(directory)


# Rename a directory or a file
def rename_instance(working_dir, old_name, new_name):
    old_path = os.path.join(working_dir, old_name)
    new_path = os.path.join(working_dir, new_name)
    os.rename(old_path, new_path)

    return new_path


# Calculate Dice, Mean Absolute Distance etc. Use comments to include/exclude metrics.
def calculate_metrics(fixed, moving):
    fixed = nib.load(fixed)
    moving = nib.load(moving)

    # Calculate Dice Similarity Coefficient
    fixed_data = np.array(fixed.get_fdata()).astype(int)
    moving_data = np.array(moving.get_fdata()).astype(int)
    ground_truth_sum = np.sum(fixed_data)
    moving_sum = np.sum(moving_data)
    intersection = fixed_data & moving_data
    intersection_sum = np.count_nonzero(intersection)
    dice_index = 2 * intersection_sum / (ground_truth_sum + moving_sum)

    # Calculate the directed Hausdorff distance between the images
    subsample = 10000
    fixed_points = np.array(np.where(fixed_data)).T
    fixed_points = fixed_points[np.random.choice(fixed_points.shape[0], subsample, replace=False)]
    moving_points = np.array(np.where(moving_data)).T
    moving_points = moving_points[np.random.choice(moving_points.shape[0], subsample, replace=False)]
    distances = cdist(fixed_points, moving_points)
    min_distances_fixed = distances.min(axis=1, initial=np.inf)
    min_distances_moving = distances.min(axis=0, initial=np.inf)
    max_distance = max(min_distances_fixed.max(), min_distances_moving.max())
    avg_distance = (min_distances_fixed.mean() + min_distances_moving.mean()) / 2.0

    return {
        "Dice": dice_index,
        "H.D Max": max_distance,
        "H.D Avg": avg_distance
    }


# Console colors.
class ConsoleColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# Image properties.
class ImageProperty:
    DIMS = 0
    TYPE = 1
    SIZE = 2
    SPACING = 3
    ORIGIN = 4
    VOXELS = 5
