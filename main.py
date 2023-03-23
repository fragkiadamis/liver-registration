"""
This script's main purpose is to create a baseline registration between an MRI and a CT scan. The input directory
containing the dicom studies is defined by the -inp flag on CLI. Before doing the registration the dicom data should be
converted to nifty images. The -out flag defines the output of the conversion. The program will automatically create
the directory if it does not exist. The data from the output directory will be used for the registration.

File execution:
$ python3 main.py -inp dicom_input_dir -out nifty_output_dir
"""

# Import necessary files and libraries
import sys
import argparse
import os
import subprocess
import shortuuid

# clitk tools binary path
clitk_tools = "C:/clitk_private/_build/bin"


# Create a respective output structure.
def create_output_structure(input_path, output_path, dir_name):
    dir_input_path = os.path.join(input_path, dir_name)
    dir_output_path = os.path.join(output_path, dir_name)
    if not os.path.isdir(dir_output_path):
        os.mkdir(dir_output_path)

    return dir_input_path, dir_output_path


# Find and return the dicom object's type (MRI, CT, REG) based on the parent's directory name
def find_obj_type(obj):
    if 'REG' in str(obj):
        return f'REG'
    elif 'MR' in str(obj):
        return f'MRI'
    elif 'CT' in str(obj):
        return f'CT'
    elif 'RTst' in str(obj):
        return f'RTst'


# Rename the directories containing the dicom files. Helps to avoid possible failure because of long paths in
# subprocess execution. Add uuid to avoid duplicate directory names.
def rename_dir(working_dir, old_dir_name, new_dir_name):
    old_path = os.path.join(working_dir, old_dir_name)
    new_path = os.path.join(working_dir, f'{new_dir_name}_{shortuuid.uuid()}')
    os.rename(old_path, new_path)

    return new_path


# Create a list from the dicom files and return the dicom series objects.
def get_dicom_series(obj_input_path):
    dcm_series = []

    # Iterate in the files of the series
    for dcm_file in os.listdir(obj_input_path):
        dcm_file_path = os.path.join(obj_input_path, dcm_file)
        dcm_series.append(str(dcm_file_path))

    return dcm_series


# Take the dicom series and cast it into a nifty file.
def dicom_series_2_nifty(obj_input_path, study_output_path, modality):
    # Get the clitkDicom2Image binary path.
    clitkDicom2Image = [os.path.join(clitk_tools, 'clitkDicom2Image')]

    # Get dicom series.
    dcm_series = get_dicom_series(obj_input_path)

    # Create command's argument list and execute.
    argument_list = ['-o', os.path.join(study_output_path, f'Volume{modality}.nii.gz'), '-t', '10']
    command = clitkDicom2Image + dcm_series + argument_list
    subprocess.run(command)


# Create nifty files from dicom files
def extract_from_dicom(dicom_dir, nifty_dir):
    # Iterate through the patients.
    for patient in os.listdir(dicom_dir):
        # Create the respective output path.
        patient_input_path, patient_output_path = create_output_structure(dicom_dir, nifty_dir, patient)

        # Iterate through the patient's studies (ceMRI, SPECT-CT, PET-CT).
        for study in os.listdir(patient_input_path):
            # Create the respective output path.
            study_input_path, study_output_path = create_output_structure(patient_input_path, patient_output_path, study)

            ascend = 0
            # Iterate through the study's objects (Image dicom series, REG file, RTStruct file).
            for obj in os.listdir(study_input_path):
                # Get type of object (MRI, CT, REG, RTStruct)
                obj_type = find_obj_type(obj)
                # Rename the objects paths to avoid any possible failures because of long paths
                obj_input_path = rename_dir(study_input_path, obj, obj_type)

                # TODO extract transformation parameters from dicom reg files.
                if obj_type == 'REG':
                    continue

                if obj_type == 'MRI' or obj_type == 'CT':
                    dicom_series_2_nifty(obj_input_path, study_output_path, obj_type)

                # TODO Dicom RTStructs to nifty.
                if obj_type == '_RTst_':
                    continue


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-inp", help="input directory")
    parser.add_argument("-out", help="output directory")

    # Upack input and output directories.
    dicom_dir, nifty_dir = vars(parser.parse_args()).values()

    # Validate that arguments are not None.
    if (not dicom_dir) or (not nifty_dir):
        sys.exit("Specify input -inp and output -out.")

    # If nifty output directory does not exist, create it
    if not os.path.isdir(nifty_dir):
        os.mkdir(nifty_dir)

    # Create nifty files from dicom files
    print('Converting dicom to nifty. This process can take several minutes...')
    extract_from_dicom(dicom_dir, nifty_dir)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
