# Import necessary files and libraries.
import os
from math import floor, ceil
from shutil import copy
from subprocess import run, check_output

import nibabel as nib
import SimpleITK as sitk
import numpy as np
from skimage.measure import label, regionprops

from utils import setup_parser, validate_paths, delete_dir, create_dir, ImageProperty


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


# Get the lower and upper bounds for padding.
def get_lower_and_upper_bounds(img, size):
    # Find minimum value to use it for padding.
    img = nib.load(img)
    img_data = np.array(img.get_fdata())
    img_min = np.min(img_data)

    lb, ub = (
        floor((size[0] - img_data.shape[0]) / 2),
        floor((size[1] - img_data.shape[1]) / 2),
        floor((size[2] - img_data.shape[2]) / 2)
    ), (
        ceil((size[0] - img_data.shape[0]) / 2),
        ceil((size[1] - img_data.shape[1]) / 2),
        ceil((size[2] - img_data.shape[2]) / 2)
    )

    return lb, ub, img_min


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


# Change the spacing and the size of the images in the pair.
def resample(pair, spacing=None, size=None):
    # Get the base image and create a list with the images that are going to be transformed according to the base image.
    base_img = pair["CT"]["volume"]
    image_list = [*[value for key, value in pair["CT"].items() if key != "volume"]]
    if "MRI" in pair:
        image_list += [value for key, value in pair["MRI"].items()]

    # Change the spacing.
    if spacing is not None:
        arg_list = [
            "clitkAffineTransform", "-i", base_img, "-o", base_img, "--interp=2",
            f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}", "--adaptive"
        ]
        run(arg_list)

    # Change the size by padding the image (can be done with clitkAffineTransform, but I want to make sure that
    # resampling and interpolation is not used in the process).
    if size is not None:
        lb, ub, img_min = get_lower_and_upper_bounds(base_img, size)
        arg_list = [
            "clitkPadImage", "-i", base_img, "-o", base_img,
            f"--lower={str(lb[0])},{str(lb[1])},{str(lb[2])}",
            f"--upper={str(ub[0])},{str(ub[1])},{str(ub[2])}",
            f"--value={img_min}"
        ]
        run(arg_list)

    # Resample the image list according to the base image.
    for img in image_list:
        image = nib.load(img)
        image_data = np.array(image.get_fdata())
        img_min = np.min(image_data)

        interp = 2 if "volume" in img else 0
        arg_list = [
            "clitkAffineTransform", "-i", img, "-o", img,
            f"--interp={interp}", "-l", base_img, f"--pad={img_min}"
        ]
        run(arg_list)


# For each mask, create a bounding box that surrounds the mask.
def create_bounding_boxes(pair):
    # Convert to nifty and save the image.
    for study in pair:
        items = {"liver": pair[study]["liver"]}
        if "tumor" in pair[study]:
            items.update({
                "tumor": pair[study]["tumor"]
            })
        if study == "CT":
            items.update({
                "unet3d_liver": pair[study]["unet3d_liver"]
            })

        for item in items:
            if item == "tumor":
                image = sitk.ReadImage(pair[study]["tumor"])
                image_arr = sitk.GetArrayFromImage(image)

                # Iterate through each slice
                for z in range(image_arr.shape[0]):
                    # Get the slice
                    slice = image_arr[z, :, :]

                    # Identify connected components
                    labeled_image = label(slice)

                    # Iterate through each connected component (region)
                    for region in regionprops(labeled_image):
                        minr, minc, maxr, maxc = region.bbox
                        # Create a bounding box around the connected component
                        image_arr[z, minr:maxr, minc:maxc] = 1

                output_img = sitk.GetImageFromArray(image_arr)
                output_img.SetSpacing(image.GetSpacing())
                output_img.SetOrigin(image.GetOrigin())
                split_path = items[item].split(".")
                sitk.WriteImage(output_img, f"{split_path[0]}_bbox.nii.gz")
            else:
                # Load the mask and convert it into a numpy array.
                mask_nii = nib.load(items[item])
                mask_data = np.array(mask_nii.get_fdata())

                # Get the segmentation_good and find min and max for each axis.
                segmentation = np.where(mask_data == 1)
                x_min, x_max = int(np.min(segmentation[0])), int(np.max(segmentation[0]))
                y_min, y_max = int(np.min(segmentation[1])), int(np.max(segmentation[1]))
                z_min, z_max = int(np.min(segmentation[2])), int(np.max(segmentation[2]))

                # Create new image and the bounding box mask.
                bounding_box_mask = np.zeros(mask_data.shape)
                bounding_box_mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1

                # Save bounding box.
                split_path = items[item].split(".")
                bounding_box_mask = bounding_box_mask.astype(np.uint8)
                bounding_box_nii = nib.Nifti1Image(bounding_box_mask, header=mask_nii.header, affine=mask_nii.affine)
                nib.save(bounding_box_nii, f"{split_path[0]}_bbox.nii.gz")


# Align MRI images to the center of gravity of CT.
def align_to_cog(pair):
    ct_label_nii = sitk.ReadImage(pair["CT"]["unet3d_liver"])
    mri_image_nii = sitk.ReadImage(pair["MRI"]["volume"])
    mri_label_nii = sitk.ReadImage(pair["MRI"]["liver"])

    ct_label_data = sitk.GetArrayFromImage(ct_label_nii)
    mri_label_data = sitk.GetArrayFromImage(mri_label_nii)
    mri_image_data = sitk.GetArrayFromImage(mri_image_nii)

    # Calculate the center of gravity for CT image and MRI image and calculate the displacement vector.
    ct_center = np.array([np.average(indices) for indices in np.where(ct_label_data == 1)])
    mri_center = np.array([np.average(indices) for indices in np.where(mri_label_data == 1)])
    displacement = ct_center - mri_center

    for item in pair["MRI"]:
        shifted_mri = mri_label_data if item == "liver" else mri_image_data
        shifted_mri = np.roll(shifted_mri, displacement.astype(int), axis=(0, 1, 2))
        shifted_mri = sitk.GetImageFromArray(shifted_mri)
        shifted_mri.SetSpacing(ct_label_nii.GetSpacing())
        shifted_mri.SetOrigin(ct_label_nii.GetOrigin())
        shifted_mri.SetDirection(ct_label_nii.GetDirection())
        sitk.WriteImage(shifted_mri, pair["MRI"][item])


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
    for patient in os.listdir(input_dir):
        print(f"-Preprocessing patient: {patient}")
        patient_dir = create_dir(output_dir, patient)

        spect_ct_volume = os.path.join(input_dir, patient, "spect_ct_volume.nii.gz")
        spect_ct_liver = os.path.join(input_dir, patient, "spect_ct_liver.nii.gz")
        spect_ct_unet3d_liver = os.path.join("./data/nii_unet3d/post_processed", f"{patient}.nii.gz")
        spect_ct_tumor = os.path.join(input_dir, patient, "spect_ct_tumor.nii.gz")
        mri_volume = os.path.join(input_dir, patient, "mri_volume.nii.gz")
        mri_liver = os.path.join(input_dir, patient, "mri_liver.nii.gz")
        mri_tumor = os.path.join(input_dir, patient, "mri_tumor.nii.gz")

        pair = {
            "CT": {
                "volume": copy(spect_ct_volume, os.path.join(output_dir, patient, "spect_ct_volume.nii.gz")),
                "liver": copy(spect_ct_liver, os.path.join(output_dir, patient, "spect_ct_liver.nii.gz")),
                "unet3d_liver": copy(spect_ct_unet3d_liver, os.path.join(patient_dir, "spect_ct_unet3d_liver.nii.gz")),
            },
            "MRI": {
                "volume": copy(mri_volume, os.path.join(output_dir, patient, "mri_volume.nii.gz")),
                "liver": copy(mri_liver, os.path.join(output_dir, patient, "mri_liver.nii.gz")),
            }
        }

        # Add conditionally the tumor path, because some RTStructs might not contain tumor labels.
        if os.path.exists(spect_ct_tumor):
            pair["CT"].update({
                "tumor": copy(spect_ct_tumor, os.path.join(output_dir, patient, "spect_ct_tumor.nii.gz"))
            })

        if os.path.exists(mri_tumor):
            pair["MRI"].update({
                "tumor": copy(mri_tumor, os.path.join(output_dir, patient, "mri_tumor.nii.gz"))
            })

        # Preprocessing for conventional registration usage.
        print(f"\t-Resampling for patient: {patient}")
        resample(pair, spacing=(1, 1, 1), size=(512, 512, 448))
        print(f"\t-Create bounding boxes for patient: {patient}")
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

        ct_volume = os.path.join(patient_dir, "spect_ct_volume.nii.gz")
        ct_liver = os.path.join(patient_dir, "spect_ct_liver.nii.gz")

        pair = {
            "CT": {
                "volume": copy(ct_volume, os.path.join(images_dir, f"{patient}.nii.gz")),
                "liver": copy(ct_liver, os.path.join(labels_dir, f"{patient}.nii.gz"))
            }
        }
        copy(ct_liver, os.path.join(unprocessed_labels_dir, f"{patient}.nii.gz"))

        pairs.append(pair)

    # Process the images.
    print("-Get median spacing...")
    median_spacing = get_prop_median(images_dir, prop=ImageProperty.SPACING)

    counter = 0
    for pair in pairs:
        print(f"-Pair {counter + 1}")
        print("\t-Resample images and labels...")
        resample(pair, spacing=median_spacing)
        print("\t-Cast images to float...")
        cast_to_type([pair["CT"]["volume"]], "float")
        print("\t-Image normalization...")
        gaussian_normalize([pair["CT"]["volume"]])

        counter += 1


# The preprocessing pipeline for pairwise deep learning registration.
def dl_reg_preprocessing(input_dir, output_dir, aligned_mri_dir=0):
    # Create the necessary structure.
    for patient in os.listdir(input_dir):
        print(f"-Preprocessing patient: {patient}")
        patient_dir = create_dir(output_dir, patient)

        spect_ct_volume = os.path.join(input_dir, patient, "spect_ct_volume.nii.gz")
        spect_ct_liver = os.path.join(input_dir, patient, "spect_ct_liver.nii.gz")
        spect_ct_unet3d_liver = os.path.join("./data/nii_unet3d/post_processed", f"{patient}.nii.gz")
        spect_ct_tumor = os.path.join(input_dir, patient, "spect_ct_tumor.nii.gz")

        if aligned_mri_dir:
            mri_volume = os.path.join(
                "./data/results/elastix/baseline_unet3d_masks", patient, "01_Affine_KS", "mri_volume_reg.nii.gz"
            )
            mri_liver = os.path.join(
                "./data/results/elastix/baseline_unet3d_masks", patient, "01_Affine_KS", "mri_liver_reg.nii.gz"
            )
            mri_tumor = os.path.join(
                "./data/results/elastix/baseline_unet3d_masks", patient, "01_Affine_KS", "mri_tumor_reg.nii.gz"
            )
        else:
            mri_volume = os.path.join(input_dir, patient, "mri_volume.nii.gz")
            mri_liver = os.path.join(input_dir, patient, "mri_liver.nii.gz")
            mri_tumor = os.path.join(input_dir, patient, "mri_tumor.nii.gz")

        pair = {
            "CT": {
                "volume": copy(spect_ct_volume, os.path.join(patient_dir, "spect_ct_volume.nii.gz")),
                "liver": copy(spect_ct_liver, os.path.join(patient_dir, "spect_ct_liver.nii.gz")),
                "unet3d_liver": copy(spect_ct_unet3d_liver, os.path.join(patient_dir, "spect_ct_unet3d_liver.nii.gz"))
            },
            "MRI": {
                "volume": copy(mri_volume, os.path.join(patient_dir, "mri_volume.nii.gz")),
                "liver": copy(mri_liver, os.path.join(patient_dir, "mri_liver.nii.gz"))
            }
        }

        if os.path.exists(spect_ct_tumor):
            pair["CT"].update({
                "tumor": copy(spect_ct_tumor, os.path.join(patient_dir, "spect_ct_tumor.nii.gz"))
            })

        if os.path.exists(spect_ct_tumor):
            pair["MRI"].update({
                "tumor": copy(mri_tumor, os.path.join(patient_dir, "mri_tumor.nii.gz"))
            })

        # Process the images.
        # print("\t-Resample images and labels...")
        # resample(pair, spacing=(2, 2, 2), size=(256, 256, 224))
        print("\t-Cast images to float...")
        cast_to_type([pair["CT"]["volume"], pair["MRI"]["volume"]], "float")
        print("\t-Image normalization...")
        gaussian_normalize([pair["CT"]["volume"], pair["MRI"]["volume"]])


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("config/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    preprocessing_type = args.t
    aligned_mri_dir = int(args.pd)
    # delete_dir(os.path.join(dir_name, args.o))
    output_dir = create_dir(dir_name, args.o)

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
