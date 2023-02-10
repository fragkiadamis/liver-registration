# Import necessary files and libraries
import sys
import argparse


# Main function of the script.
def main():
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
