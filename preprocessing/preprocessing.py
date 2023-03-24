# Import necessary files and libraries.
import os
from dotenv import load_dotenv
import shortuuid
from utils import \
    validate_paths, \
    create_output_structure, \
    rename_instance, \
    get_files, \
    fix_filenames, \
    execute_shell_cmd


# Find and return the dicom object's type (MRI, CT, REG) based on the parent's directory name.
def find_obj_type(obj):
    if "REG" in str(obj):
        return "REG"
    elif "MR" in str(obj):
        return "MRI"
    elif "CT" in str(obj):
        return "CT"
    elif "RTst" in str(obj):
        return "RTst"


# Take the dicom series and convert it to nifty.
def dicom_series_2_nifty(obj_input_path, study_output_path, modality):
    # Get dicom series files.
    dcm_series = get_files(obj_input_path)
    # Volume output path + filename.
    volume_path = os.path.join(study_output_path, f"{modality}_Volume.nii.gz")

    # Create command and execute.
    execute_shell_cmd("clitkDicom2Image", [*dcm_series, "-o", volume_path, "-t", "10"])

    # Return Volume's path to use the volume later for the RT structure conversion.
    return volume_path


# Take the dicom RT Structures and convert them to nifty.
def dicom_rtst_2_nifty(obj_input_path, study_output_path, volume_path):
    # Get dicom files (RT Structs are inside 1 file).
    dcm_rtstr = get_files(obj_input_path)[0]
    # RT structure output path + file basename.
    rtstr_path = os.path.join(study_output_path, "RTStruct")

    # Create command and execute.
    execute_shell_cmd("clitkDicomRTStruct2Image", ["-i", dcm_rtstr, "-j", volume_path, "-o", rtstr_path,  "--niigz", "-t", "10"])


# Create nifty files from dicom files.
def extract_from_dicom(dicom_dir, nifty_dir):
    patients = []

    # Iterate through the patients.
    for patient in os.listdir(dicom_dir):
        print(f"\n-Extracting from patient: {patient}")
        # Create the respective output path.
        patient_input_path, patient_output_path = create_output_structure(dicom_dir, nifty_dir, patient)

        # Iterate through the patient's studies (ceMRI, SPECT-CT, PET-CT).
        for study in os.listdir(patient_input_path):
            print(f"\t-Study: {study}")
            # Create the respective output path.
            study_input_path, study_output_path = create_output_structure(patient_input_path, patient_output_path, study)

            volume_path = ""
            # Iterate through the study's objects (Image dicom series, REG file, RTStruct file).
            for obj in os.listdir(study_input_path):
                print(f"\t\t-Object: {obj}")
                # Get type of object (MRI, CT, REG, RTStruct).
                obj_type = find_obj_type(obj)
                # Rename the objects paths to avoid any possible failures because of long paths.
                # TODO: Check sometime in the future if it is possible to handle large strings in the subprocess and
                #  completely delete the renaming process from the project.
                obj_input_path = rename_instance(study_input_path, obj, f"{obj_type}_{shortuuid.uuid()}")

                # TODO extract transformation parameters from dicom reg files.
                if obj_type == "REG":
                    continue

                if obj_type == "MRI" or obj_type == "CT":
                    volume_path = dicom_series_2_nifty(obj_input_path, study_output_path, obj_type)

                if obj_type == "RTst":
                    dicom_rtst_2_nifty(obj_input_path, study_output_path, volume_path)

                # Fix any inconsistencies in the filenames
                fix_filenames(study_output_path)

        patients.append(patient)

    print("Extraction done.")
    return patients


# Resample MRI instance to the CT instance's physical space.
def resample_mri_2_ct(ct_mri_pairs):
    for pair in ct_mri_pairs:
        print(f"{pair} pair resampling...")
        ct_instance = ct_mri_pairs[pair][0]
        mri_instance = ct_mri_pairs[pair][1]

        split_path = mri_instance.split(".")
        resampled_mri_struct = f"{split_path[0]}_resampled_to_ct.{split_path[1]}.{split_path[2]}"
        execute_shell_cmd("clitkAffineTransform", ["-i", mri_instance, "-o", resampled_mri_struct, "-l", ct_instance])

        ct_mri_pairs[pair][1] = resampled_mri_struct

    print("Resampling Done.")
    return ct_mri_pairs


# TODO: Add the function from Felix' preprocessing file (the one in the USB key) that locates identical or very
#  similar patients.


def main():
    # Load environmental variables from .env file (create a .env file according to .env.example).
    load_dotenv()

    dicom_dir = os.environ["DICOM_PATH"]
    nifty_dir = os.environ["NIFTY_PATH"]
    validate_paths(dicom_dir, nifty_dir)

    # Convert dicom data to nifty and extract the manual transformations.
    extract_from_dicom(dicom_dir, nifty_dir)
    print("Extraction done.")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
