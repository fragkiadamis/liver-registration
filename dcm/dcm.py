# Import necessary files and libraries.
import os
from utils import parse_arguments
import subprocess
import shortuuid


# Create a respective output structure.
def create_output_structure(input_path, output_path, dir_name):
    dir_input_path = os.path.join(input_path, dir_name)
    dir_output_path = os.path.join(output_path, dir_name)
    if not os.path.isdir(dir_output_path):
        os.mkdir(dir_output_path)

    return dir_input_path, dir_output_path


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


# Rename the directories containing the dicom files. Helps to avoid possible failure because of long paths in
# subprocess execution. Add uuid to avoid duplicate directory names.
def rename_dir(working_dir, old_dir_name, new_dir_name):
    old_path = os.path.join(working_dir, old_dir_name)
    new_path = os.path.join(working_dir, f"{new_dir_name}_{shortuuid.uuid()}")
    os.rename(old_path, new_path)

    return new_path


# Create a list from the files and return the dicom series objects.
def get_files(obj_input_path):
    dcm_series = []

    # Iterate in the files of the series.
    for dcm_file in os.listdir(obj_input_path):
        dcm_file_path = os.path.join(obj_input_path, dcm_file)
        dcm_series.append(str(dcm_file_path))

    return dcm_series


# Create shell command and execute it.
def execute_shell_cmd(cmd, arguments):
    clitk_tools = os.environ.get("CLITK_TOOLS_PATH")
    clitk_command = [os.path.join(clitk_tools, cmd)]
    command = clitk_command + arguments
    subprocess.run(command)


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
                obj_input_path = rename_dir(study_input_path, obj, obj_type)

                # TODO extract transformation parameters from dicom reg files.
                if obj_type == "REG":
                    continue

                if obj_type == "MRI" or obj_type == "CT":
                    volume_path = dicom_series_2_nifty(obj_input_path, study_output_path, obj_type)

                if obj_type == "RTst":
                    dicom_rtst_2_nifty(obj_input_path, study_output_path, volume_path)

    print("Extraction done.")


def main():
    # Parse CLI arguments and handle inputs and outputs.
    dicom_dir, nifty_dir = parse_arguments()

    # Convert dicom data to nifty and extract the manual transformations.
    extract_from_dicom(dicom_dir, nifty_dir)
    print("Extraction done.")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
