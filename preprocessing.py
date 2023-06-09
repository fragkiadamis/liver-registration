# Import necessary files and libraries.
import os
from shutil import copytree, copy
from subprocess import run, check_output

import nibabel as nib
import numpy as np

from utils import setup_parser, validate_paths, delete_dir, create_dir


# Traverse through the given dataset paths and create paired paths between the available modalities.
def get_pair_paths(parent_dir, studies):
    pair = {}
    for study in studies:
        pair[f"{study if 'CT' in study else 'MRI'}"] = {
            "volume": os.path.join(parent_dir, study, "volume.nii.gz"),
            "liver": os.path.join(parent_dir, study, "liver.nii.gz"),
        }
    return pair


# Get the spacing for each axis of the image.
def get_spacing(input_image):
    output = check_output(["clitkImageInfo", input_image])
    output = output.decode()
    spacing = output.split(" ")[3]
    spacing_splitted = spacing.split("x")

    # Return the spacing in float type.
    return [float(x) for x in spacing_splitted]


# Find and return the minimum spacing for each axis between two volumes.
def find_minimum_spacing(path_1, path_2):
    img_1_spacing = get_spacing(path_1)
    img_2_spacing = get_spacing(path_2)

    spacing = (
        min(img_1_spacing[0], img_2_spacing[0]),
        min(img_1_spacing[1], img_2_spacing[1]),
        min(img_1_spacing[2], img_2_spacing[2])
    )
    return spacing


# Calculate and return the median spacing of the dataset.
def get_median_spacing(input_dir):
    # Get the spacing for all the images.
    spacing_list = []
    for image in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image)
        image_spacing = get_spacing(image_path)
        spacing_list.append(image_spacing)

    # Get the median from the list.
    spacing_list = np.asarray(spacing_list)
    median_spacing = np.median(spacing_list, axis=0)

    return median_spacing


# Perform a bias field correction for the MRIs.
def bias_field_correction(mri):
    run(["clitkN4BiasFieldCorrection", "-i", mri, "-o", mri])


# Change the spacing of the images in the list.
def resample(images, spacing):
    ct_images, mri_images = images["CT"], images["MRI"]
    for img in ct_images:
        interp = 2 if img == "volume" else 0
        arg_list = ["clitkAffineTransform", "-i", ct_images[img], "-o", ct_images[img], f"--interp={interp}",
                    f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}", "--adaptive"]
        run(arg_list)

    for img in mri_images:
        interp = 2 if img == "volume" else 0
        arg_list = ["clitkAffineTransform", "-i", mri_images[img], "-o", mri_images[img],
                    f"--interp={interp}", "-l", ct_images[img]]
        run(arg_list)


# For each mask, create a boundary box that surrounds the mask.
def create_bounding_boxes(pair):
    # Convert to nifty and save the image.
    for study in pair:
        liver = pair[study]["liver"]

        # Load the mask and convert it into a numpy array.
        mask_nii = nib.load(liver)
        mask_data = np.array(mask_nii.get_fdata())

        # Get the segmentation_good and find min and max for each axis.
        segmentation = np.where(mask_data == 1)
        x_min, x_max = int(np.min(segmentation[0])), int(np.max(segmentation[0]))
        y_min, y_max = int(np.min(segmentation[1])), int(np.max(segmentation[1]))
        z_min, z_max = int(np.min(segmentation[2])), int(np.max(segmentation[2]))

        # Create new image and the bounding box mask.
        bounding_box_mask = np.zeros(mask_data.shape)
        bounding_box_mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1

        # Save bounding boxes.
        split_path = liver.split(".")
        bounding_box_mask = bounding_box_mask.astype(np.uint8)
        bounding_box_nii = nib.Nifti1Image(bounding_box_mask, header=mask_nii.header, affine=mask_nii.affine)
        nib.save(bounding_box_nii, f"{split_path[0]}_bb.{split_path[1]}.gz")


# Cast the voxel type of the images into float.
def cast_to_type(image_paths, img_type):
    for img_path in image_paths:
        run(["clitkImageConvert", "-i", img_path, "-o", img_path, "-t", img_type])


# Get the statistics of the image.
def get_statistics(img_path):
    output = check_output(["clitkImageStatistics", '-v', img_path])
    output = output.decode()
    output_split = output.split("\n")

    mean = float(output_split[9][5:])
    std = 1 / float(output_split[10][3:])

    return -mean, std


# Perform a gaussian normalization to the images.
def gaussian_normalize(image_paths):
    for img_path in image_paths:
        stats = get_statistics(img_path)

        for idx, stat in enumerate(stats):
            run(["clitkImageArithm", "-i", img_path, "-o", img_path, "-s", str(stat), "-t", str(idx)])


# Perform a min/max normalization to the images (0 - 1).
def min_max_normalization(image_paths):
    for img_path in image_paths:
        run(["clitkNormalizeImageFilter", "-i", img_path, "-o", img_path])


# The preprocessing pipeline for data use from the elastix software.
def elastix_preprocessing(input_dir, output_dir):
    # Create the respective output structures. Those will be used for registration.
    delete_dir(output_dir)
    copytree(input_dir, output_dir)

    # Preprocessing for conventional registration usage.
    for patient in os.listdir(output_dir):
        patient_dir = os.path.join(output_dir, patient)

        pairs = {
            "CT": {
                "volume": os.path.join(patient_dir, "ct_volume.nii.gz"),
                "liver": os.path.join(patient_dir, "ct_liver.nii.gz"),
            },
            "MRI": {
                "volume": os.path.join(patient_dir, "mri_volume.nii.gz"),
                "liver": os.path.join(patient_dir, "mri_liver.nii.gz"),
            }
        }

        print(f"-Bias field correction for patient: {patient}")
        bias_field_correction(pairs["MRI"]["volume"])
        print(f"-Resampling for patient: {patient}")
        resample(pairs, (1, 1, 1))
        print(f"-Create boundary boxes for patient: {patient}")
        create_bounding_boxes(pairs)


# The preprocessing pipeline for ct scan deep learning segmentation.
def dl_seg_preprocessing(input_dir, output_dir):
    training_set = {
        "images": create_dir(f"{output_dir}/segmentation", "images"),
        "labels": create_dir(f"{output_dir}/segmentation", "labels"),
        "unprocessed_labels": create_dir(f"{output_dir}/segmentation", "non_resampled_labels")
    }

    for patient in os.listdir(input_dir):
        patient_dir = os.path.join(input_dir, patient)

        ct_volume = os.path.join(patient_dir, "ct_volume.nii.gz")
        ct_liver = os.path.join(patient_dir, "ct_liver.nii.gz")

        copy(ct_volume, os.path.join(training_set["images"], f"{patient}_ct_volume.nii.gz"))
        copy(ct_liver, os.path.join(training_set["labels"], f"{patient}_ct_liver.nii.gz"))
        copy(ct_liver, os.path.join(training_set["unprocessed_labels"], f"{patient}_ct_liver.nii.gz"))

    image_paths = [os.path.join(training_set["images"], path) for path in os.listdir(training_set["images"])]
    label_paths = [os.path.join(training_set["labels"], path) for path in os.listdir(training_set["labels"])]

    print("Get median spacing...")
    median_spacing = get_median_spacing(training_set["images"])
    print("Resample images...")
    resample(image_paths, median_spacing)
    print("Resample labels...")
    resample(label_paths, median_spacing)
    print("Cast images to float...")
    cast_to_type(image_paths, "float")
    print("Image normalization...")
    gaussian_normalize(image_paths)
    # min_max_normalization(image_paths)


# The preprocessing pipeline for pairwise deep learning registration.
def dl_reg_preprocessing(input_dir, output_dir, prealligned_mri, prealligned_mri_dir):
    training_set = {
        "images": create_dir(f"{output_dir}/registration", "images"),
        "labels": create_dir(f"{output_dir}/registration", "labels"),
        "unprocessed_labels": create_dir(f"{output_dir}/registration", "non_resampled_labels")
    }

    for patient in os.listdir(input_dir):
        if prealligned_mri:
            ct_volume = os.path.join(input_dir, patient, "ct_volume.nii.gz")
            ct_label = os.path.join(input_dir, patient, "ct_liver.nii.gz")
            mri_volume = os.path.join(prealligned_mri_dir, patient, "01_Affine_KS", "mri_volume_reg.nii.gz")
            mri_label = os.path.join(prealligned_mri_dir, patient, "01_Affine_KS", "mri_liver_reg.nii.gz")
        else:
            ct_volume = os.path.join(input_dir, patient, "ct_volume.nii.gz")
            ct_label = os.path.join(input_dir, patient, "ct_liver.nii.gz")
            mri_volume = os.path.join(input_dir, patient, "mri_volume.nii.gz")
            mri_label = os.path.join(input_dir, patient, "mri_liver.nii.gz")

        copy(ct_volume, os.path.join(training_set["images"], f"{patient}_fixed.nii.gz"))
        copy(ct_label, os.path.join(training_set["labels"], f"{patient}_fixed.nii.gz"))
        copy(mri_volume, os.path.join(training_set["images"], f"{patient}_moving.nii.gz"))
        copy(mri_label, os.path.join(training_set["labels"], f"{patient}_moving.nii.gz"))


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("config/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)
    preprocessing_type = args.t
    prealligned_mri = int(args.prealigned)
    prealligned_mri_dir = os.path.join(dir_name, args.d)

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    if preprocessing_type == "elx":
        elastix_preprocessing(input_dir, output_dir)
    elif preprocessing_type == "dls":
        dl_seg_preprocessing(input_dir, output_dir)
    elif preprocessing_type == "dlr":
        dl_reg_preprocessing(input_dir, output_dir, prealligned_mri, prealligned_mri_dir)
    else:
        print("Provide a valid type for preprocessing.")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
