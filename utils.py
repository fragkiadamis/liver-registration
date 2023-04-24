import argparse
import json
import os
import sys

from shutil import rmtree, copytree


# Setup the parser and the correct argument messages loaded from the respective json file.
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


# Create the respective output structures 2 levels deep.
def create_output_structures(input_dir, output_dir, depth=2, identical=False):
    # Reset the output directory.
    delete_dir(output_dir)

    if identical:
        copytree(input_dir, output_dir)
        return

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


# Rename a directory or a file
def rename_instance(working_dir, old_name, new_name):
    old_path = os.path.join(working_dir, old_name)
    new_path = os.path.join(working_dir, new_name)
    os.rename(old_path, new_path)

    return new_path


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
