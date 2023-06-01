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
            "tumor": os.path.join(parent_dir, study, "tumor.nii.gz")
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
        print(f"\t-CT {img}")
        run(arg_list)

    for img in mri_images:
        interp = 2 if img == "volume" else 0
        arg_list = ["clitkAffineTransform", "-i", mri_images[img], "-o", mri_images[img],
                    f"--interp={interp}", "-l", ct_images[img]]
        print(f"\t-MRI {img}")
        run(arg_list)


# Crop the pairs. For each study, crop automatically the mask and based on the mask, crop the volume.
def crop(pair):
    for study in pair:
        print(f"\t-Study: {study}")

        # Crop the liver mask automatically.
        liver_path = pair[study]["liver"]

        print(f"\t\t-Nifty: {liver_path}")
        run(["clitkAutoCrop", "-i", liver_path, "-o", liver_path])

        # Crop the volume and the tumor mask according to the cropped liver mask.
        volume_path = pair[study]["volume"]
        tumor_path = pair[study]["tumor"]

        print(f"\t\t-Nifty: {volume_path}")
        run(["clitkCropImage", "-i", volume_path, "--like", liver_path, "-o", volume_path])
        print(f"\t\t-Nifty: {tumor_path}")
        run(["clitkCropImage", "-i", tumor_path, "--like", liver_path, "-o", tumor_path])


# For each mask, create a boundary box that surrounds the mask.
def create_bounding_boxes(pair):
    for study in pair:
        print(f"\t-Study: {study}")
        volume = pair[study]["volume"]
        for mask in [pair[study]["liver"], pair[study]["tumor"]]:
            print(f"\t\t-Mask: {mask}")
            # Load the mask and convert it into a numpy array.
            mask_nii = nib.load(mask)
            mask_data = np.array(mask_nii.get_fdata())

            # Get the segmentation and find min and max for each axis.
            segmentation = np.where(mask_data == 1)
            x_min, x_max = int(np.min(segmentation[0])), int(np.max(segmentation[0]))
            y_min, y_max = int(np.min(segmentation[1])), int(np.max(segmentation[1]))
            z_min, z_max = int(np.min(segmentation[2])), int(np.max(segmentation[2]))

            # Create new image and the bounding box mask.
            bounding_box_mask = np.zeros(mask_data.shape)
            bounding_box_mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1

            # Convert to nifty and save the image.
            split_mask = mask.split(".")
            bounding_box_mask = bounding_box_mask.astype(np.uint8)
            bounding_box_mask = nib.Nifti1Image(bounding_box_mask, header=mask_nii.header, affine=mask_nii.affine)
            nib.save(bounding_box_mask, f"{split_mask[0]}_bb.{split_mask[1]}.gz")

        print(f"\t\t-Volume: {volume}")
        # Load the volume of the study and the liver bounding box mask.
        volume_nii = nib.load(volume)
        volume_data = np.array(volume_nii.get_fdata())

        # Get the liver bounding box.
        split = volume.split("volume")
        liver_bb_mask = nib.load(f"{split[0]}liver_bb{split[1]}")
        liver_bb_mask_data = np.array(liver_bb_mask.get_fdata())

        # Create a new volume and keep only the intensities inside the liver's bounding box.
        split_volume = volume.split(".")
        bounding_box_image = volume_data * liver_bb_mask_data
        bounding_box_image = bounding_box_image.astype(np.ushort)
        bounding_box_image = nib.Nifti1Image(bounding_box_image, header=volume_nii.header, affine=volume_nii.affine)
        nib.save(bounding_box_image, f"{split_volume[0]}_bb.{split_volume[1]}.gz")


# Cast the voxel type of the images into float.
def cast_to_type(image_paths, type):
    for img_path in image_paths:
        run(["clitkImageConvert", "-i", img_path, "-o", img_path, "-t", type])


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
        print(f"Normalising {img_path}")
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
                "tumor": os.path.join(patient_dir, "ct_tumor.nii.gz")
            },
            "MRI": {
                "volume": os.path.join(patient_dir, "mri_volume.nii.gz"),
                "liver": os.path.join(patient_dir, "mri_liver.nii.gz"),
                "tumor": os.path.join(patient_dir, "mri_tumor.nii.gz")
            }
        }

        # print(f"-Bias field correction for patient: {patient}")
        # bias_field_correction(pairs["MRI"]["volume"])
        print(f"-Resampling for patient: {patient}")
        resample(pairs, (1, 1, 1))
        # print(f"\n-Cropping for patient: {patient}")
        # crop(pair)
        print(f"-Create boundary boxes for patient: {patient}")
        create_bounding_boxes(pairs)
        print()


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
