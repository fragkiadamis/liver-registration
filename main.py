"""
This script's main purpose is to create a baseline registration between an MRI and a CT scan. The input directory
containing the dicom studies is defined by the -i flag on CLI. Before doing the registration the dicom data should be
converted to nifty images. The -o flag defines the output of the conversion. The program will automatically create
the directory if it does not exist. The data from the output directory will be used for the registration.

File execution:
$ python3 main.py -i dicom_input_dir -o nifty_output_dir
"""

# Import necessary files and libraries
import sys
import argparse


def main():
    # Print docstring of file
    print(__doc__)

    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input directory")
    parser.add_argument("-o", help="output directory")
    args = parser.parse_args()

    # Validate that arguments are not None
    if (not args.i) or (not args.o):
        sys.exit("Specify input -i and output -o.")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
