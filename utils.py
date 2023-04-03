import os
import sys
from shutil import rmtree
import pandas as pd


# Validate paths, create output paths if they do not exist
def validate_paths(input_dir, output_dir):
    # Validate that arguments are not None.
    if (not input_dir) or (not output_dir):
        sys.exit("Specify input -inp, output -out and processed output -pout.")

    # Validate that input directory exists.
    if not os.path.exists(input_dir):
        sys.exit("This input directory does not exist.")


# Delete the defined directory if it exists and create it again.
def delete_directory(directory):
    if os.path.exists(directory):
        rmtree(directory)
    os.mkdir(directory)


# Create a directory.
def create_dir(output_path, dir_name):
    dir_output_path = os.path.join(output_path, dir_name)
    if not os.path.exists(dir_output_path):
        os.mkdir(dir_output_path)

    return dir_output_path


# Create the respective output structures 2 levels deep.
def create_output_structures(input_dir, output_dir):
    delete_directory(output_dir)

    # Create 1st level directories.
    for item in os.listdir(input_dir):
        item_output = create_dir(output_dir, item)

        # Create 2nd level directories.
        sub_dir = os.path.join(input_dir, item)
        for sub_item in os.listdir(sub_dir):
            create_dir(item_output, sub_item)


# Create a dataframe and load the xl file if it exists.
def open_data_frame(file):
    df = pd.DataFrame()
    # Open the file with all the data (if it exists).
    if file in os.listdir("./"):
        df = pd.read_excel(file, index_col=0)

    return df


# Add a new column to the dataframe.
def update_dataframe(df, data, column_name):
    for item in data:
        df.loc[df["Patient"] == item, column_name] = data[item]
    # Save the data
    df.to_excel("output.xlsx")
    return df


# Get all the image paths forall the studies for all the patients.
def get_dataset(root_dir):
    dataset = {}
    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)

        dataset[patient] = {}
        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)

            dataset[patient][study] = {
                "volume": os.path.join(study_path, f"{study}_volume.nii.gz"),
                "rtst_liver": os.path.join(study_path, f"{study}_rtstruct_liver.nii.gz"),
                "rtst_tumor": os.path.join(study_path, f"{study}_rtstruct_tumor.nii.gz")
            }

    return dataset


# Traverse through the given dataset paths and create paired paths between the available modalities.
def create_paired_paths(parent_dir, studies):
    pairs = {}
    for study in studies:
        pairs[f"{'CT' if 'CT' in study else 'MRI'}"] = {
            "volume": os.path.join(parent_dir, study, f"{study}_volume.nii.gz"),
            "rtstruct_liver": os.path.join(parent_dir, study, f"{study}_rtstruct_liver.nii.gz"),
            "rtstruct_tumor": os.path.join(parent_dir, study, f"{study}_rtstruct_tumor.nii.gz")
        }
    return pairs
