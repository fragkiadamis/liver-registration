# Import necessary files and libraries.
import os
from subprocess import run

# from pydicom import read_file
# import h5py
import nibabel as nib
import numpy as np

from utils import setup_parser, validate_paths, rename_instance, create_dir, delete_dir


# Return a list with the files of the directory.
def get_files(path):
    files = []
    # Iterate in the files of the series.
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        files.append(str(file_path))

    return files


# Add an increasing number at the end of the name if the name already exists in a directory.
def fix_duplicate(parent_dir, name):
    ascending = 1
    contents = [x.split(".")[0] for x in os.listdir(parent_dir)]

    # If already exists, change name again...
    while name in contents:
        if name[-1].isnumeric():
            name = name[:-1] + str(ascending)
        else:
            name = name + f"{ascending}"
        ascending += 1

    return name


# Rename the directories inside the dicom directory.
def rename_dicom_structure(study_input):
    studies = os.listdir(study_input)
    for idx, std_name in enumerate(studies):
        std_name_new = \
            "MR_DEF" if "Def" in std_name and "_MR_" in std_name else \
            "MR" if "_MR_" in std_name else \
            "CT" if "_CT_" in std_name else \
            "RTst_DEF" if "def" in std_name and "_RTst_" in std_name else \
            "RTst" if "_RTst_" in std_name else \
            "REG" if "_REG_" in std_name else ""
        std_name_new = fix_duplicate(study_input, std_name_new)
        std_dir = rename_instance(study_input, std_name, std_name_new)

        for dicom_file in os.listdir(std_dir):
            # Take the file ascending number and the file format (XXXX.XXXX.{...}.XXXX.dcm -> XXXX.dcm).
            name_split = (dicom_file.split("."))[-2:]
            new_name = f"{name_split[0]}.{name_split[1]}"
            rename_instance(std_dir, dicom_file, new_name)


# Lower filenames and fix any inconsistency (e.g. foie -> liver).
def fix_filenames(patient_dir, modality):
    segmentations = []
    for filename in os.listdir(patient_dir):
        # Only the RT Structs of the current modality need fixing.
        if "volume" in filename or modality not in filename:
            continue

        new_filename = filename.lower()
        # Make the RT structures filename consistent. If it's a necrosis segmentation or necrosis related, delete it.
        if "necrosis" in new_filename:
            # new_filename = f"{modality}_necrosis.nii.gz"
            print(f"\t\t\t-Deleting file: {filename}")
            os.remove(os.path.join(patient_dir, filename))
            continue

        elif "foie" in new_filename or "liver" in new_filename:
            new_filename = f"{modality}_liver.nii.gz"

        elif "tumeur" in new_filename or "tumor" in new_filename or "tum" in new_filename:
            new_filename = f"{modality}_tumor.nii.gz"

        split_name = new_filename.split(".")
        base_file_name = split_name[0]
        base_file_name = fix_duplicate(patient_dir, base_file_name)
        new_filename = f"{base_file_name}.nii.gz"
        print(f"\t\t\t-Renaming file: {filename} ---> {new_filename}")
        rename_instance(patient_dir, filename, new_filename)

        if f"{modality}_tumor" in new_filename:
            segmentations.append(new_filename)

    return segmentations


# Create one segmentation by adding all the different segmentations of the specified anatomy.
def add_segmentations(segmentations, volume_path, modality, patient_dir):
    # Get the respective volume information.
    volume_nii = nib.load(volume_path)
    volume_size = np.array(volume_nii.get_fdata()).shape

    # Initiate the total_roi array and iterate through the segmentations to add them.
    total_roi = np.zeros(volume_size)
    for roi in segmentations:
        roi_path = os.path.join(patient_dir, roi)
        roi_nii = nib.load(roi_path)
        total_roi += np.array(roi_nii.get_fdata())

    # Make sure that the binary image values are [0, 1]
    # If everything went well, delete all the fragmented tumor ROI files.
    total_roi[total_roi > 1] = 1
    if len(np.unique(total_roi)) > 2:
        print(f"Warning: {np.unique(np.unique(total_roi))}")
    else:
        [os.remove(os.path.join(patient_dir, roi)) for roi in segmentations]

    # Save the total roi.
    total_roi = total_roi.astype(np.uint8)
    total_roi = nib.Nifti1Image(total_roi, header=volume_nii.header, affine=volume_nii.affine)
    nib.save(total_roi, os.path.join(patient_dir, f"{modality}_tumor.nii.gz"))


# Extract the nifty images from the DICOM series and the RT Structures as well as the transformation parameters from
# the REG files from the manual registration.
def extract_from_dicom(study_input, patient_output):
    # Before proceeding, Rename directories and files to avoid long names.
    # Long names might cause problems with the windows command prompt.
    rename_dicom_structure(study_input)

    dicom_series = [x for x in os.listdir(study_input) if x == "MR" or x == "MR_DEF" or x == "CT"]
    dicom_rtstr = [x for x in os.listdir(study_input) if x == "RTst" or x == "RTst_DEF"]

    # Extract the volume from the series in nifty format.
    volume_path = ""
    modality = "mri_def" if "ceMRI-DEF" in study_input else "mri" if "ceMRI" in study_input else "ct"

    for series in dicom_series:
        series_path = os.path.join(study_input, series)
        series_files = get_files(series_path)
        volume_path = os.path.join(patient_output, f"{modality}_volume.nii.gz")
        print(f"\t\t-Extract volume: {series} ---> {volume_path}")
        run(["clitkDicom2Image", *series_files, "-o", volume_path, "-t", "10"])

    # Extract the masks from the RT structures in nifty format. Use the extracted volume from above.
    for i, rt_structure in enumerate(dicom_rtstr):
        rtstruct_path = os.path.join(study_input, rt_structure)
        rtstruct_file = get_files(rtstruct_path)[0]
        print(f"\t\t-Extract RT struct: {rt_structure} ---> {patient_output}/")
        run(
            ["clitkDicomRTStruct2Image", "-i", rtstruct_file, "-j", volume_path,
             "-o", f"{patient_output}/{modality}_", "--niigz", "-t", "10"]
        )

        # Fix the filenames of the rois and return the tumor segmentations in the last run.
        if i == len(dicom_rtstr) - 1:
            tumor_rois = fix_filenames(patient_output, modality)
            # If the tumor is segmented in multiple files, add the rois in one file.
            if len(tumor_rois) > 1:
                print(f"\t\t-Adding the fragmented tumor segmentations into one file.")
                add_segmentations(tumor_rois, volume_path, modality, patient_output)

    # TODO: Make sure that the correct information is being extracted.
    # for registration in dicom_reg:
    #     registration_path = os.path.join(study_input, registration)
    #     registration_file = get_files(registration_path)[0]
    #
    #     print(f"\t\t-Extract transform: {registration}")
    #     dcm_reg = read_file(registration_file)
    #     vector_grid = np.asarray(dcm_reg.DeformableRegistrationSequence[-1].
    #                              DeformableRegistrationGridSequence[-1][0x64, 0x09].value)
    #     reg_matrix = np.asarray(dcm_reg.DeformableRegistrationSequence[-1].
    #                             PreDeformationMatrixRegistrationSequence[-1][0x3006, 0xc6].value)
    #
    #     print(dir(dcm_reg.DeformableRegistrationSequence[-1]))
    #
    #     file = open(f"{study_output}/transformation.mat", "w+")
    #     file.write(str(reg_matrix))
    #     file.close()
    #     h5f = h5py.File(f"{study_output}/transformation.h5", "w")
    #     h5f.create_dataset('dataset_1', data=reg_matrix)


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
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/database_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    delete_dir(os.path.join(dir_name, args.o))
    output_dir = create_dir(dir_name, args.o)

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # For each patient of the dataset.
    for patient in os.listdir(input_dir):
        print(f"\n-Extracting patient: {patient}")
        patient_input = os.path.join(input_dir, patient)
        patient_output = create_dir(output_dir, patient)

        # For each study of the patient (ceMRI, SPECT-CT).
        for study in os.listdir(patient_input):
            if study == "PET-CT":
                continue

            print(f"\t-Study: {study}")
            study_input = os.path.join(patient_input, study)
            extract_from_dicom(study_input, patient_output)

    # # Make a check to handle any possible duplicate data.
    # check_for_duplicates(output_dir, os.listdir(output_dir))


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
