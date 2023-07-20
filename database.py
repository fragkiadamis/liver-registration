# Import necessary files and libraries.
import os
from struct import unpack
from subprocess import run, check_output

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pydicom import read_file

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


# Create one segmentation by adding all the different segmentations of the specified anatomy.
def add_segmentations(segmentations, volume_path, patient_dir):
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
    new_filename = "total_tumor.nii.gz"
    total_roi = total_roi.astype(np.uint8)
    total_roi = nib.Nifti1Image(total_roi, header=volume_nii.header, affine=volume_nii.affine)
    nib.save(total_roi, os.path.join(patient_dir, new_filename))

    return new_filename


# Lower filenames and fix any inconsistency (e.g. foie -> liver).
def fix_filenames(patient_dir, extracted_files, modality, volume_path):
    print(f"\t\t\t-Extracted Files: {extracted_files}")

    # Delete necrosis related files (not necessary).
    [
        (print(f"\t\t\t-Deleting file: {file}"), os.remove(os.path.join(patient_dir, file)))
        for file in extracted_files if "necrosis" in file.lower()
    ]

    # Separate liver & tumor segmentations.
    liver_files = [file for file in extracted_files if "liver" in file.lower() or "foie" in file.lower()]
    tumor_files = [
        file for file in extracted_files
        if "tumor" in file.lower() or \
           "tumeur" in file.lower() or \
           ("tum" in file.lower() and "necrosis" not in file.lower())
    ]

    # Fix the liver filenames, distinguish between the liver and the registered mri contours.
    for file in liver_files:
        if "ct" in modality and ("mri" in file.lower() or "irm" in file.lower()):
            print(f"\t\t\t-Deleting file: {file}")
            os.remove(os.path.join(patient_dir, file))
        else:
            liver_filename = f"{modality}_liver.nii.gz"
            print(f"\t\t\t-Renaming file: {file} ---> {liver_filename}")
            rename_instance(patient_dir, file, liver_filename)

    # Separate registered MRI tumors and CT tumor delineations.
    tumors_reg = [file for file in tumor_files if "ct" in modality and ("mri" in file.lower() or "irm" in file.lower())]
    tumors = [file for file in tumor_files if file not in tumors_reg]

    # Handle the tumor files.
    def handle_tumors(tumor_list):
        tumor_filename = f"{modality}_tumor.nii.gz"
        if len(tumor_list) > 1:
            print(f"\t\t\t-Adding the segmentations: {tumor_list}")
            file = add_segmentations(tumor_list, volume_path, patient_dir)
        else:
            file = tumor_list[0]
        print(f"\t\t\t-Renaming file: {file} ---> {tumor_filename}")
        rename_instance(patient_dir, file, tumor_filename)

    # Handle CT tumor delineations.
    if len(tumors) > 0:
        # If there are available tumor delineations of the modality delete the registered from the MRI.
        for tumor_file in tumors_reg:
            print(f"\t\t\t-Deleting file: {tumor_file}")
            os.remove(os.path.join(patient_dir, tumor_file))
        handle_tumors(tumors)

    # If there are no CT tumor delineations keep the MRI registered tumor.
    elif len(tumors_reg) > 0 and len(tumors) == 0:
        handle_tumors(tumors_reg)


# Extract global registration.
def get_global_transform(dcm_reg, seq):
    # Get registration info.
    reg_info = None
    if seq == "pre":
        reg_info = dcm_reg["DeformableRegistrationSequence"][-1]["PreDeformationMatrixRegistrationSequence"][0]
    elif seq == "post":
        reg_info = dcm_reg["DeformableRegistrationSequence"][-1]["PostDeformationMatrixRegistrationSequence"][0]

    # Extract transformation matrix.
    transformation_matrix = np.asarray(reg_info.FrameOfReferenceTransformationMatrix)
    # Reshape it to remove last row and re-flatten it.
    transformation_matrix = transformation_matrix.reshape((4, 4))[:3, :3]
    transformation_matrix = sitk.VectorDouble(transformation_matrix.flatten())

    # Create transform with the transformation matrix.
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(transformation_matrix)

    # And return it.
    return transform


# Extract deformable registration.
def get_deformable_transform(dcm_reg):
    # Get registration info.
    reg_info = dcm_reg["DeformableRegistrationSequence"][-1]["DeformableRegistrationGridSequence"][0]

    # Extract grid buffer and convert to numbers.
    dvf_vals_raw = reg_info.VectorGridData
    dvf_vals = unpack(f"<{len(dvf_vals_raw) // 4}f", dvf_vals_raw)

    # Extract grid properties.
    dvf_grid_size = reg_info["GridDimensions"].value
    dvf_grid_res = reg_info["GridResolution"].value
    dvf_grid_origin = reg_info["ImagePositionPatient"].value

    # Create a grid object.
    dvf_arr = np.reshape(dvf_vals, dvf_grid_size[::-1] + [3, ])
    dvf_img = sitk.GetImageFromArray(dvf_arr)
    dvf_img.SetSpacing(dvf_grid_res)
    dvf_img.SetOrigin(dvf_grid_origin)

    # convert it to a transform and return it.
    return sitk.DisplacementFieldTransform(dvf_img)


# Extract the nifty images from the DICOM series and the RT Structures as well as the transformation parameters from
# the REG files from the manual registration.
def extract_from_dicom(study_input, patient_output):
    # Before proceeding, Rename directories and files to avoid long names.
    # Long names might cause problems with the windows command prompt.
    rename_dicom_structure(study_input)

    dicom_series = [x for x in os.listdir(study_input) if x == "MR" or x == "MR_DEF" or x == "CT"]
    dicom_rtstr = [x for x in os.listdir(study_input) if x == "RTst" or x == "RTst_DEF"]
    # dicom_reg = [x for x in os.listdir(study_input) if x == "REG"]

    # Extract the volume from the series in nifty format.
    volume_path = ""
    modality = None
    if "ceMRI-DEF" in study_input:
        modality = "mri_manual_reg"
    elif "ceMRI" in study_input:
        modality = "mri"
    elif "SPECT-CT" in study_input:
        modality = "spect_ct"
    elif "PET-CT" in study_input:
        modality = "pet_ct"

    for series in dicom_series:
        series_path = os.path.join(study_input, series)
        series_files = get_files(series_path)
        volume_name = f"{modality}_volume.nii.gz"
        volume_path = os.path.join(patient_output, volume_name)
        print(f"\t\t-Extract volume: {series} ---> {volume_name}")
        run(["clitkDicom2Image", *series_files, "-o", volume_path, "-t", "10"])

    # Extract the masks from the RT structures in nifty format. Use the extracted volume from above.
    for rt_structure in dicom_rtstr:
        rtstruct_path = os.path.join(study_input, rt_structure)
        rtstruct_file = get_files(rtstruct_path)[0]
        print(f"\t\t-Extract RT struct: {rt_structure} ---> {patient_output}/")
        out = check_output(
            ["clitkDicomRTStruct2Image", "-i", rtstruct_file, "-j", volume_path,
             "-o", f"{patient_output}/", "--niigz", "-t", "10", "-v"]
        )

        extracted_files = []
        lines = out.decode().split("\n")
        for line in lines:
            if line:
                file = line.rstrip().split("/")[-1]
                extracted_files.append(file)

        # Fix the filenames of the rois and return the tumor segmentations in the last run.
        fix_filenames(patient_output, extracted_files, modality, volume_path)

    # for registration in dicom_reg:
    #     study = study_input.split("/")[-1]
    #     if study != "ceMRI":
    #         break
    #
    #     registration_path = os.path.join(study_input, registration)
    #     registration_file = get_files(registration_path)[0]
    #
    #     print(f"\t\t-Extract transform: {registration}")
    #     dcm_reg = read_file(registration_file)
    #
    #     mri_items = {}
    #     mri_items.update({
    #         item.split(".")[0].split("_")[1]: item for item in os.listdir(patient_output) if item.startswith("mri_")
    #     })
    #
    #     ct_volume_path = os.path.join(patient_output, "spect_ct_volume.nii.gz")
    #
    #     for item in mri_items:
    #         if "reg" in mri_items[item]:
    #             continue
    #
    #         # inp = os.path.join(patient_output, mri_items[item])
    #         # run(["clitkAffineTransform", "-i", inp, "-o", inp, "-l", f"{ct_volume_path}"])
    #         # Read the image.
    #         item_path = os.path.join(patient_output, mri_items[item])
    #         item_img = sitk.ReadImage(item_path)
    #         item_data = sitk.GetArrayFromImage(item_img)
    #         item_min = float(np.min(item_data))
    #
    #         # item_img.SetSpacing(ct_volume.GetSpacing())
    #         # item_img.SetOrigin(ct_volume.GetOrigin())
    #         # item_img.SetDirection(ct_volume.GetDirection())
    #
    #         # Set up the interpolator.
    #         interpolator = sitk.sitkBSpline if item == "volume" else sitk.sitkNearestNeighbor
    #
    #         # Perform registration.
    #         pre_deformable_reg = get_global_transform(dcm_reg, seq="pre")
    #         transformed = sitk.Resample(
    #             item_img, pre_deformable_reg, interpolator=interpolator,
    #             defaultPixelValue=item_min, outputPixelType=sitk.sitkUInt16
    #         )
    #
    #         deformable_transform = get_deformable_transform(dcm_reg)
    #         transformed = sitk.Resample(
    #             transformed, deformable_transform, interpolator=interpolator,
    #             defaultPixelValue=item_min, outputPixelType=sitk.sitkUInt16
    #         )
    #
    #         post_deformable_reg = get_global_transform(dcm_reg, seq="post")
    #         transformed = sitk.Resample(
    #             transformed, post_deformable_reg, interpolator=interpolator,
    #             defaultPixelValue=item_min, outputPixelType=sitk.sitkUInt16
    #         )
    #         sitk.WriteImage(transformed, f"{patient_output}/mri_{item}_reg.nii.gz")


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
    # delete_dir(os.path.join(dir_name, args.o))
    output_dir = create_dir(dir_name, args.o)

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # For each patient of the dataset.
    for i, patient in enumerate(os.listdir(input_dir)):
        print(f"\n-{i + 1} Extracting patient: {patient}")
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
