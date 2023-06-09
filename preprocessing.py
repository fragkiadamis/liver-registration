# Import necessary files and libraries.
import os
from shutil import copytree
from subprocess import run, check_output

import nibabel as nib
import numpy as np

from utils import setup_parser, validate_paths, delete_dir


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


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("config/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

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
        print()


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
