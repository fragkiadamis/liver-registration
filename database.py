# Import necessary files and libraries.
import os
from subprocess import run
import numpy as np
import nibabel as nib
from utils import setup_parser, validate_paths, create_output_structures, rename_instance


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
        if filename == "volume.nii.gz":
            return

        new_filename = filename.lower()

        # Make the RT structures filename consistent.
        if "foie" in new_filename or "liver" in new_filename:
            new_filename = "liver.nii.gz"
        elif "tumeur" in new_filename or "tumor" in new_filename:
            new_filename = "tumor.nii.gz"
        elif "necrosis" in new_filename:
            new_filename = "necrosis.nii.gz"

        # If already exists, change name again...
        ascending = 1
        while os.path.exists(os.path.join(parent_dir, new_filename)):
            split_path = new_filename.split(".")
            new_filename = f"{split_path[0]}{ascending}.{split_path[1]}.{split_path[2]}"
            ascending += 1

        rename_instance(parent_dir, filename, new_filename)


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
        volume_path = os.path.join(study_output, "volume.nii.gz")
        print(f"\t\t-Extract volume: {series} ---> {volume_path}")
        run(["clitkDicom2Image", *series_files, "-o", volume_path, "-t", "10"])

    # Extract the masks from the RT structures in nifty format. Use the extracted volume from above.
    for rt_structure in dicom_rtstr:
        rtstruct_path = os.path.join(study_input, rt_structure)
        rtstruct_file = get_files(rtstruct_path)[0]
        print(f"\t\t-Extract RT struct: {rtstruct_path} ---> {study_output}")
        run(
            ["clitkDicomRTStruct2Image", "-i", rtstruct_file, "-j", volume_path,
             "-o", f"{study_output}/", "--niigz", "-t", "10"]
        )
        fix_filenames(study_output, study)

    # TODO: extract transformation parameters from dicom reg files (not in priority).
    for registration in dicom_reg:
        print(f"\t\t-Extract transform: {registration}")
        continue


# Compare the two volumes two see if they are exact the same or similar.
def compare_volumes(path_a, path_b):
    volume_a, volume_b = nib.load(path_a).get_fdata(), nib.load(path_b).get_fdata()
    volume_a, volume_b = np.array(volume_a, dtype=np.int32), np.array(volume_b, dtype=np.int32)
    shape_a, shape_b = np.shape(volume_a), np.shape(volume_b)

    result = 2
    if shape_a == shape_b:
        volume_sub = volume_a - volume_b
        summation = np.sum(np.absolute(volume_sub))

        if summation == 0:
            result = -1
        elif summation < 10000:
            result = 0
        elif summation < 100000:
            result = 1

    return result


# Check for duplicate patients in the dataset. Exact duplicates will be removed automatically, very similar ones are
# going to be stored in the duplicates directory and will be handled manually by the user. handled manually by the user.
def check_for_duplicates(input_dir, patients):
    for patient_a in patients:
        print(f"-Checking for duplicates for {patient_a}")
        patient_a_path = os.path.join(input_dir, patient_a)
        for study in os.listdir(patient_a_path):
            print(f"\t-Checking study {study}")
            volume_a_path = os.path.join(patient_a_path, str(study), "volume.nii.gz")

            # Remove self and check on the rest of the patients.
            list_without_self = patients.copy()
            list_without_self.remove(patient_a)
            for patient_b in list_without_self:
                print(f"\t\t-Against patient {patient_b}")
                patient_b_path = os.path.join(input_dir, patient_b)
                volume_b_path = os.path.join(patient_b_path, str(study), "volume.nii.gz")

                print(f"\t\t\t-Comparing {volume_a_path} with {volume_b_path}")
                result = compare_volumes(volume_a_path, volume_b_path)
                if result == -1:
                    print("\t\t\t-These images are exactly the same")
                elif result == 0:
                    print("\t\t\t-These images might be the same patient")
                elif result == 1:
                    print("\t\t\t-These images look alike")
                elif result == 2:
                    print("\t\t\t-These images seem OK!")


def main():
    args = setup_parser("messages/database_parser.json")
    # Set arguments.
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

    # Make a check to handle any possible duplicate data.
    check_for_duplicates(input_dir, os.listdir(input_dir))


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
