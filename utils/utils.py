import sys
import os
from fnmatch import fnmatch
from subprocess import check_output


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


# TODO: Create a function that creates the dicom input structure and copies the files from the inconsistent dataset
#  structure
def create_structure():
    """"""


# TODO: The nifty output structure should be a copy of the the dicom input structure (maybe with a layer less deep).
# Create the output structure.
def create_output_structure(input_path, output_path, dir_name):
    dir_input_path = os.path.join(input_path, dir_name)
    dir_output_path = os.path.join(output_path, dir_name)
    if not os.path.isdir(dir_output_path):
        os.mkdir(dir_output_path)

    return dir_input_path, dir_output_path


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
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        files.append(str(file_path))

    return files


# Lower filenames and fix any inconsistency (e.g. foie -> liver).
def fix_filenames(path):
    for filename in os.listdir(path):
        new_filename = filename.lower()

        # Make the RT structures filename consistent.
        if "foie" in new_filename or "liver" in new_filename:
            new_filename = "rtstruct_liver.nii.gz"
        elif "tumeur" in new_filename or "tumor" in new_filename:
            new_filename = "rtstruct_tumor.nii.gz"

        rename_instance(path, filename, new_filename)


# Traverse through the data and find the defined studies (ceMRI, SPECT-CT, PET-CT) pair according to filename
def find_ct_mri_pair(root_dir, studies, filename):
    pairs = {}

    for patient in os.listdir(root_dir):
        study_one_dir = os.path.join(root_dir, patient, studies[0])
        study_two_dir = os.path.join(root_dir, patient, studies[1])

        # Find and get the liver structure for the MRI and the CT
        instance_one, instance_two, match = "", "", ""

        for nii_file in os.listdir(study_one_dir):
            if fnmatch(nii_file, f"*{filename}.nii.gz"):
                instance_one = os.path.join(study_one_dir, nii_file)

        for nii_file in os.listdir(study_two_dir):
            if fnmatch(nii_file, f"*{filename}.nii.gz"):
                instance_two = os.path.join(study_two_dir, nii_file)

        pairs[patient] = [instance_one, instance_two]

    return pairs


# Create shell command and execute it.
def execute_shell_cmd(cmd, arguments):
    clitk_command = [os.path.join(os.environ["CLITK_TOOLS_PATH"], cmd)]
    command = clitk_command + arguments

    return check_output(command)
