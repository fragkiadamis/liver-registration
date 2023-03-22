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


# Create nifty files from dicom files
def dicom_2_nifty(dicom_dir, nifty_dir):
    # Iterate through the dicom studies.
    for patient_dir in os.listdir(dicom_dir):
        patient_input_path = os.path.join(dicom_dir, patient_dir)
        patient_output_path = os.path.join(nifty_dir, patient_dir)

        # Create the output directory for each study.
        if not os.path.isdir(patient_output_path):
            os.mkdir(patient_output_path)

        # For each modality in the study (ceMRI, SPECT-CT, PET-CT)
        for study in os.listdir(patient_input_path):
            # Get path for object input and create the respective output path
            study_path = os.path.join(patient_input_path, study)
            output_study_path = os.path.join(patient_output_path, study)
            os.mkdir(output_study_path)

            # For each modality in the study (Image dicom series, REG file, RTStruct file)
            for obj in os.listdir(study_path):
                # We do not care about REG files in this process
                if '_REG_' in obj:
                    continue

                # Cast to nifty the modality dicom series
                if '_MR_' in obj or '_CT_' in obj:
                    print(obj)


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
    dicom_2_nifty(dicom_dir, nifty_dir)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
