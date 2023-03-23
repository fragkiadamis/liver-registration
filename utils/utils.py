import argparse
import sys
import os


# Validate paths, create output paths if they do not exist
def validate_paths(input_dir, output_dir):
    # Validate that arguments are not None.
    if (not input_dir) or (not output_dir):
        sys.exit("Specify input -inp and output -out.")

    # Validate that input directory exists.
    if not os.path.isdir(input_dir):
        sys.exit("This input directory does not exist.")

    # Create output if it does not exist.
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


# Parse the input and the output arguments from the cli and handle any errors.
def parse_arguments():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-inp", help="input directory")
    parser.add_argument("-out", help="output directory")

    input_dir, output_dir = vars(parser.parse_args()).values()

    # Return input and output directories.
    return input_dir, output_dir
