import argparse
import os
from subprocess import run
from utils import validate_paths, create_output_structures


# Return a list with the files of the directory.
def get_files(path):
    files = []
    # Iterate in the files of the series.
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        files.append(str(file_path))

    return files


# Lower filenames and fix any inconsistency (e.g. foie -> liver).
def fix_filenames(parent_dir, study):
    for filename in os.listdir(parent_dir):
        # Only the RT Structs need fixing.
        if filename == f"{study}_volume.nii.gz":
            return

        new_filename = filename.lower()

        # Make the RT structures filename consistent.
        if "foie" in new_filename or "liver" in new_filename:
            new_filename = f"{study}_rtstruct_liver.nii.gz"
        elif "tumeur" in new_filename or "tumor" in new_filename:
            new_filename = f"{study}_rtstruct_tumor.nii.gz"

        # If already exists, change name again...
        ascending = 1
        while os.path.exists(os.path.join(parent_dir, new_filename)):
            split_path = new_filename.split(".")
            new_filename = f"{split_path[0]}_{ascending}.{split_path[1]}.{split_path[2]}"
            ascending += 1

        rename_instance(parent_dir, filename, new_filename)


# Rename a directory
def rename_instance(working_dir, old_name, new_name):
    old_path = os.path.join(working_dir, old_name)
    new_path = os.path.join(working_dir, new_name)
    os.rename(old_path, new_path)

    return new_path


# Extract the nifty images from the DICOM series and the RT Structures as well as the transformation parameters from
# the REG files from the manual registration.
def extract_from_dicom(study_input, study_output, study):
    # Separate the series, the RT structs and the registration files.
    dicom_series = [x for x in os.listdir(study_input) if "_MR_" in x or "_CT_" in x]
    dicom_rtstr = [x for x in os.listdir(study_input) if "_RTst_" in x]
    dicom_reg = [x for x in os.listdir(study_input) if "_REG_" in x]

    # Before proceeding, Rename files to avoid long names (might cause problems with the windows command prompt).
    for dicom in (dicom_series + dicom_rtstr + dicom_reg):
        dicom_dir = os.path.join(study_input, dicom)
        for dicom_file in os.listdir(dicom_dir):
            # Take the file ascending number and the file format (XXXX.XXXX.{...}.XXXX.dcm -> XXXX.dcm).
            name_split = (dicom_file.split("."))[-2:]
            new_name = f"{name_split[0]}.{name_split[1]}"
            rename_instance(dicom_dir, dicom_file, new_name)

    # Extract the volume from the series in nifty format.
    volume_path = ""
    for series in dicom_series:
        series_path = os.path.join(study_input, series)
        series_files = get_files(series_path)
        volume_path = os.path.join(study_output, f"{study}_volume.nii.gz")
        print(f"\t\t-Extract series: {series} ---> {volume_path}")
        run(["clitkDicom2Image", *series_files, "-o", volume_path, "-t", "10"])

    # Extract the masks from the RT structures in nifty format. Use the extracted volume from above.
    for rt_structure in dicom_rtstr:
        rtstruct_path = os.path.join(study_input, rt_structure)
        rtstruct_file = get_files(rtstruct_path)[0]
        rtst_basename = os.path.join(study_output, f"{study}_rtstruct")
        print(f"\t\t-Extract RT struct: {rtstruct_path} ---> {rtst_basename}.nii.gz")
        run(
            ["clitkDicomRTStruct2Image", "-i", rtstruct_file, "-j", volume_path,
             "-o", rtst_basename, "--niigz", "-t", "10"]
        )
        fix_filenames(study_output, study)

    # TODO extract transformation parameters from dicom reg files.
    for registration in dicom_reg:
        print(f"\t\t-DICOM: {registration}")
        continue


def main():
    parser = argparse.ArgumentParser(description="Create the image databases from the raw database")
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-i", help="path to the directory where is the dicom database", required=True)
    required_args.add_argument("-o", help="path to a directory where to save the nifty database", required=True)
    args = parser.parse_args()

    input_dir, output_dir = args.i, args.o

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir, depth=2)

    # For each patient of the dataset.
    for patient in os.listdir(input_dir):
        print(f"\n-Extracting patient: {patient}")
        patient_input = os.path.join(input_dir, patient)
        patient_output = os.path.join(output_dir, patient)

        # For each study of the patient (ceMRI, SPECT-CT, PET-CT).
        for study in os.listdir(patient_input):
            print(f"\t-Study: {study}")
            study_input = os.path.join(patient_input, study)
            study_output = os.path.join(patient_output, study)
            extract_from_dicom(study_input, study_output, study)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
