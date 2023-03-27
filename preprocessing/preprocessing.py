# Import necessary files and libraries.
import os
from dotenv import load_dotenv
from subprocess import check_output
from utils import \
    validate_paths, \
    create_dir, \
    get_files, \
    rename_instance, \
    fix_filenames


# Create nifty files from dicom files.
def extract_from_dicom(dicom_dir, nifty_dir):
    patients = []

    # Iterate through the patients.
    for patient in os.listdir(dicom_dir):
        print(f"\n-Extracting from patient: {patient}")
        # Create the respective output path.
        patient_input_path = os.path.join(dicom_dir, patient)
        patient_output_path = create_dir(nifty_dir, patient)

        # Iterate through the patient's studies (ceMRI, SPECT-CT, PET-CT).
        for study in os.listdir(patient_input_path):
            print(f"\t-Study: {study}")
            # Create the respective output path.
            study_input_path = os.path.join(patient_input_path, study)
            study_output_path = create_dir(patient_output_path, study)

            volume_path = ""
            # Iterate through the study's objects (Image dicom series, REG file, RTStruct file).
            for obj in os.listdir(study_input_path):
                print(f"\t\t-Object: {obj}")
                obj_input_path = os.path.join(study_input_path, str(obj))
                # Rename files to avoid long names (might cause problems with the windows command prompt).
                for dcm_file in os.listdir(obj_input_path):
                    # Take the file ascending number and the file format. All the file names are the same but the
                    # ascending number at the end.
                    name_split = (dcm_file.split("."))[-2:]
                    new_name = f"{name_split[0]}.{name_split[1]}"
                    rename_instance(obj_input_path, dcm_file, new_name)

                # TODO extract transformation parameters from dicom reg files.
                if "REG" in obj:
                    continue

                if "MR" in obj or "CT" in obj:
                    dcm_path = get_files(obj_input_path)
                    volume_path = os.path.join(study_output_path, f"Volume.nii.gz")
                    check_output(["clitkDicom2Image", *dcm_path, "-o", volume_path, "-t", "10"])

                if "RTst" in obj_input_path:
                    dcm_path = get_files(obj_input_path)[0]
                    rest_path = os.path.join(study_output_path, "rtstruct")
                    check_output(
                        ["clitkDicomRTStruct2Image", "-i", dcm_path, "-j", volume_path,
                         "-o", rest_path, "--niigz", "-t", "10"]
                    )

                # Fix any inconsistencies in the filenames
                fix_filenames(study_output_path)
        patients.append(patient)
    print("Extraction done.")
    return patients


# Resample MRI instance to the CT instance's physical space.
def resample_mri_2_ct(ct_mri_pairs):
    for patient in ct_mri_pairs:
        print(f"{patient} pair resampling...")
        ct_rtst_instance = ct_mri_pairs[patient][1]
        mri_rtst_instance = ct_mri_pairs[patient][3]

        split_path = mri_rtst_instance.split(".")
        resampled_mri_struct = f"{split_path[0]}_resampled_to_ct.{split_path[1]}.{split_path[2]}"

        check_output(["clitkAffineTransform", "-i", mri_rtst_instance, "-o", resampled_mri_struct, "-l", ct_rtst_instance])

        ct_mri_pairs[patient][3] = resampled_mri_struct

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
