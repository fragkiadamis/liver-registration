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
def create_output_structures(input_dir, output_dir, depth):
    delete_directory(output_dir)

    # Create 1st level directories.
    for item in os.listdir(input_dir):
        item_output = create_dir(output_dir, item)

        if depth == 1:
            continue

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
