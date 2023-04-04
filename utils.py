import argparse
import json
import os
import sys
from shutil import rmtree
import pandas as pd


# Setup the parser and the correct argument messages loaded from the respective json file.
def setup_parser(json_file):
    with open(json_file, 'r') as messages:
        messages = json.load(messages)

        parser = argparse.ArgumentParser(description=messages["description"])
        required_args = parser.add_argument_group("required arguments")

        for arg in messages["required"]:
            message = messages["required"][arg]
            required_args.add_argument(arg, help=message, required=True)

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


# Create the respective output structures 2 levels deep.
def create_output_structures(input_dir, output_dir, depth):
    # Reset the directory.
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)

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

    # Delete previous output if it exists.
    if file in os.listdir("./"):
        os.remove(file)

    return df


# Add a new column to the dataframe.
def update_dataframe(df, patient, value, column_name):
    df.loc[patient, column_name] = value
    df.to_excel("output.xlsx")
    return df


# Calculate average for each column of the dataframe.
def dataframe_averages(df):
    df.loc['Average'] = df.mean()
    df.to_excel("output.xlsx")
