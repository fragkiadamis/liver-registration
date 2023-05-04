import os
from shutil import rmtree
from utils import setup_parser


# Read the arguments from the parser, find the specified experiments by iterating in the registration output
# directory and finally delete the directories of these experiments.
def main():
    args = setup_parser("parser/delete_experiment_parser.json")
    elastix_dir, experiments = args.i, args.exp.split(",")

    for exp in experiments:
        if exp in os.listdir(elastix_dir):
            rmtree(os.path.join(elastix_dir, exp))


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
