import os
from shutil import rmtree
from utils import setup_parser


# Read the arguments from the parser, find the specified experiments by iterating in the registration output
# directory and finally delete the directories of these experiments.
def main():
    args = setup_parser("parser/delete_experiment_parser.json")
    parent_dir, experiments = args.i, args.exp.split(",")

    patients = os.listdir(parent_dir)
    for patient in patients:
        patient_path = os.path.join(parent_dir, patient)
        for exp in experiments:
            if exp in os.listdir(patient_path):
                rmtree(os.path.join(patient_path, exp))


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
