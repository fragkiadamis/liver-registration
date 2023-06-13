# Import necessary files and libraries.
import os
from shutil import copytree, copy
from subprocess import run, check_output

import nibabel as nib
import numpy as np

from utils import setup_parser, validate_paths, delete_dir, create_dir, ImageProperty


# Traverse through the given dataset paths and create paired paths between the available modalities.
def get_pair_paths(parent_dir, studies):
    pair = {}
    for study in studies:
        pair[f"{study if 'CT' in study else 'MRI'}"] = {
            "volume": os.path.join(parent_dir, study, "volume.nii.gz"),
            "liver": os.path.join(parent_dir, study, "liver.nii.gz"),
        }
    return pair


# Calculate and return the median spacing of the dataset.
def get_prop_median(input_dir, prop):
    # Get the spacing for all the images.
    prop_list = []
    for image in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image)

        output = check_output(["clitkImageInfo", image_path])
        output = output.decode()
        spacing = output.split(" ")[prop]
        spacing_splitted = spacing.split("x")

        prop_list.append([float(x) for x in spacing_splitted])

    # Get the median from the list.
    prop_list = np.asarray(prop_list)
    median_spacing = np.median(prop_list, axis=0)

    return median_spacing


# Perform a bias field correction for the MRIs.
def bias_field_correction(mri):
    run(["clitkN4BiasFieldCorrection", "-i", mri, "-o", mri])


# Change the spacing and the size of the images in the pair.
def resample(pair, spacing=None, size=None, adaptive=True):
    # Get the base image and create a list with the images that are going to be transformed according to the base image.
    base_img = pair["CT"]["volume"]
    image_list = [*[value for key, value in pair["CT"].items() if key != "volume"]]
    if "MRI" in pair:
        image_list += [value for key, value in pair["MRI"].items()]

    # FInd minimum value to use it for padding.
    ct_image = nib.load(base_img)
    ct_image_data = np.array(ct_image.get_fdata())
    ct_min = np.min(ct_image_data)

    # Change the spacing.
    if spacing:
        arg_list = [
            "clitkAffineTransform", "-i", base_img, "-o", base_img, "--interp=2",
            f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}", f"--pad={ct_min}"
        ]
        if adaptive:
            arg_list.append("--adaptive")
        run(arg_list)

    # Change the size.
    if size:
        arg_list = [
            "clitkAffineTransform", "-i", base_img, "-o", base_img, "--interp=2",
            f"--size={str(size[0])},{str(size[1])},{str(size[2])}", f"--pad={ct_min}"
        ]
        if adaptive:
            arg_list.append("--adaptive")
        run(arg_list)

    # Resample the image list according to the base image.
    for img in image_list:
        interp = 2 if "volume" in img else 0
        arg_list = ["clitkAffineTransform", "-i", img, "-o", img, f"--interp={interp}", "-l", base_img]
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

    for patient in os.listdir(output_dir):
        patient_dir = os.path.join(output_dir, patient)

        pair = {
            "CT": {
                "volume": os.path.join(patient_dir, "ct_volume.nii.gz"),
                "liver": os.path.join(patient_dir, "ct_liver.nii.gz"),
            },
            "MRI": {
                "volume": os.path.join(patient_dir, "mri_volume.nii.gz"),
                "liver": os.path.join(patient_dir, "mri_liver.nii.gz"),
            }
        }

        # Preprocessing for conventional registration usage.
        print(f"-Bias field correction for patient: {patient}")
        bias_field_correction(pair["MRI"]["volume"])
        print(f"-Resampling for patient: {patient}")
        resample(pair, spacing=(1, 1, 1))
        print(f"-Create boundary boxes for patient: {patient}")
        create_bounding_boxes(pair)


# The preprocessing pipeline for ct scan deep learning segmentation.
def dl_seg_preprocessing(input_dir, output_dir):
    images_dir = create_dir(f"{output_dir}", "images")
    labels_dir = create_dir(f"{output_dir}", "labels")
    unprocessed_labels_dir = create_dir(f"{output_dir}", "non_resampled_labels")

    # Create the necessary structure.
    pairs = []
    for patient in os.listdir(input_dir):
        patient_dir = os.path.join(input_dir, patient)

        ct_volume = os.path.join(patient_dir, "ct_volume.nii.gz")
        ct_liver = os.path.join(patient_dir, "ct_liver.nii.gz")

        pair = {
            "CT": {
                "volume": copy(ct_volume, os.path.join(images_dir, f"{patient}.nii.gz")),
                "liver": copy(ct_liver, os.path.join(labels_dir, f"{patient}.nii.gz"))
            }
        }
        copy(ct_liver, os.path.join(unprocessed_labels_dir, f"{patient}.nii.gz"))

        pairs.append(pair)

    # Process the images.
    print("Get median spacing...")
    median_spacing = get_prop_median([*[pair["CT"]["volume"] for pair in pairs]], prop=ImageProperty.SPACING)

    counter = 0
    for pair in pairs:
        print(f"-Pair {counter}")
        print("\t-Resample images and labels...")
        resample(pair, spacing=median_spacing)
        print("\t-Cast images to float...")
        cast_to_type([pair["CT"]["volume"]], "float")
        print("\t-Image normalization...")
        gaussian_normalize([pair["CT"]["volume"]])

        counter += 1


# The preprocessing pipeline for pairwise deep learning registration.
def dl_reg_preprocessing(input_dir, output_dir, aligned_mri_dir=None):
    # Create the necessary structure.
    for patient in os.listdir(input_dir):
        print(f"-Processing {patient}")
        patient_dir = create_dir(output_dir, patient)

        ct_volume = os.path.join(input_dir, patient, "ct_volume.nii.gz")
        ct_label = os.path.join(input_dir, patient, "ct_liver.nii.gz")

        if aligned_mri_dir:
            mri_volume = os.path.join(aligned_mri_dir, patient, "01_Affine_KS", "mri_volume_reg.nii.gz")
            mri_label = os.path.join(aligned_mri_dir, patient, "01_Affine_KS", "mri_liver_reg.nii.gz")
        else:
            mri_volume = os.path.join(input_dir, patient, "mri_volume.nii.gz")
            mri_label = os.path.join(input_dir, patient, "mri_liver.nii.gz")

        pair = {
            "CT": {
                "volume": copy(ct_volume, os.path.join(patient_dir, "ct_volume.nii.gz")),
                "liver": copy(ct_label, os.path.join(patient_dir, "ct_liver.nii.gz"))
            },
            "MRI": {
                "volume": copy(mri_volume, os.path.join(patient_dir, "mri_volume.nii.gz")),
                "liver": copy(mri_label, os.path.join(patient_dir, "mri_liver.nii.gz"))
            }
        }

        # Process the images.
        print("\t-Resample images and labels...")
        resample(pair, spacing=(2, 2, 2), size=(256, 256, 128), adaptive=False)
        print("\t-Cast images to float...")
        cast_to_type([pair["CT"]["volume"], pair["MRI"]["volume"]], "float")
        print("\t-Image normalization...")
        gaussian_normalize([pair["CT"]["volume"], pair["MRI"]["volume"]])


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("config/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)
    preprocessing_type = args.t
    aligned_mri_dir = os.path.join(dir_name, args.pd) if args.pd else None

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    if preprocessing_type == "elx":
        elastix_preprocessing(input_dir, output_dir)
    elif preprocessing_type == "dls":
        dl_seg_preprocessing(input_dir, output_dir)
    elif preprocessing_type == "dlr":
        dl_reg_preprocessing(input_dir, output_dir, aligned_mri_dir=aligned_mri_dir)
    else:
        print("Provide a valid type for preprocessing.")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
