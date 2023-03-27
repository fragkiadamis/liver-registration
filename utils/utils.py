import sys
import os
from subprocess import check_output
import pandas as pd


# Validate paths, create output paths if they do not exist
def validate_paths(input_dir, output_dir):
    # Validate that arguments are not None.
    if (not input_dir) or (not output_dir):
        sys.exit("Specify input -inp and output -out.")

    # Validate that input directory exists.
    if not os.path.exists(input_dir):
        sys.exit("This input directory does not exist.")

    # Create output if it does not exist.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


# Create a directory.
def create_dir(output_path, dir_name):
    dir_output_path = os.path.join(output_path, dir_name)
    if not os.path.exists(dir_output_path):
        os.mkdir(dir_output_path)

    return dir_output_path


# Rename a directory
def rename_instance(working_dir, old_name, new_name):
    old_path = os.path.join(working_dir, old_name)
    new_path = os.path.join(working_dir, new_name)
    os.rename(old_path, new_path)

    return new_path


# Return a list with the files of the directory.
def get_files(path):
    files = []
    # Iterate in the files of the series.
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        files.append(str(file_path))

    return files


# Lower filenames and fix any inconsistency (e.g. foie -> liver).
def fix_filenames(parent_dir):
    for filename in os.listdir(parent_dir):
        new_filename = filename.lower()

        # If already exists, leave it as it is...
        if os.path.exists(os.path.join(parent_dir, new_filename)):
            return

        # Make the RT structures filename consistent.
        if "foie" in new_filename or "liver" in new_filename:
            new_filename = "rtstruct_liver"
        elif "tumeur" in new_filename or "tumor" in new_filename:
            new_filename = "rtstruct_tumor"

        rename_instance(parent_dir, filename, new_filename)


# Create a dataframe and load the xl file if it exists.
def open_data_frame(file):
    df = pd.DataFrame()
    # Open the file with all the data (if it exists).
    if "output.xlsx" in os.listdir("./"):
        df = pd.read_excel("output.xlsx", index_col=0)

    return df


# Add a new column to the dataframe.
def update_dataframe(df, data, column_name):
    for item in data:
        df.loc[df["Patient"] == item, column_name] = data[item]
    # Save the data
    df.to_excel("output.xlsx")
    return df


# TODO: Set this function to return either the masks only or the images only or both masks and images
# Traverse through the data and find the defined studies (ceMRI, SPECT-CT, PET-CT) pair according to filename
def find_ct_mri_pairs(root_dir, studies):
    pairs = {}

    for patient in os.listdir(root_dir):
        study_one_dir = os.path.join(root_dir, patient, studies[0])
        study_two_dir = os.path.join(root_dir, patient, studies[1])

        pairs[patient] = [
            os.path.join(study_one_dir, "ct_volume.nii.gz"),
            os.path.join(study_one_dir, "rtstruct_liver.nii.gz"),
            os.path.join(study_two_dir, "mri_volume.nii.gz"),
            os.path.join(study_two_dir, "rtstruct_liver.nii.gz"),
        ]

    return pairs


# Calculate the dice index between 2 RT structures
def calculate_dice(rt_structs):
    dice_results = {}

    for patient in rt_structs:
        ct_struct = rt_structs[patient][1]
        mri_struct = rt_structs[patient][3]
        dice_index = check_output(["clitkDice", "-i", ct_struct, "-j", mri_struct])
        dice_index = float(dice_index.decode("utf-8"))

        print(f"{patient} Dice Index: {dice_index}")
        dice_results[f"{patient}"] = dice_index

    return dice_results
