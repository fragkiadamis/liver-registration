# Import necessary files and libraries.
import os
from subprocess import run, check_output

import nibabel as nib
import numpy as np

from utils import setup_parser, validate_paths, create_output_structures


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


# Create one segmentation by adding all the different segmentations of the specified anatomy.
def add_segmentations(patient_dir, anatomies):
    for anatomy in anatomies:
        for study in anatomies[anatomy]:
            study_dir = os.path.join(patient_dir, study)
            segmentations, size, volume_nii = [], [], None

            for file in os.listdir(study_dir):
                file_path = os.path.join(study_dir, file)

                if "volume" in file:
                    volume_nii = nib.load(file_path)
                    size = np.array(volume_nii.get_fdata()).shape

                if anatomy in file:
                    segmentations.append(file_path)

            total_seg = np.zeros(size)
            for seg_path in segmentations:
                segmentation = nib.load(seg_path)
                total_seg += np.array(segmentation.get_fdata())
                os.remove(seg_path)

            total_seg = total_seg.astype(np.uint8)
            total_seg = nib.Nifti1Image(total_seg, header=volume_nii.header, affine=volume_nii.affine)
            nib.save(total_seg, os.path.join(study_dir, "tumor.nii.gz"))


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


# Bring all the images of a patient into the same physical space of the chosen one.
def resample_moving_2_fixed(pair, like_modality):
    # Get the image that gives the physical space reference.
    like_img = pair[like_modality]["volume"]
    # Get the images of the study that are about to be resampled.
    study = "CT" if like_modality == "MRI" else "MRI"
    images = pair[study]

    print(f"\t-Bring {study} in the physical space of {like_modality}")
    # Resample the images and store them in the respective output directory.
    for image in images:
        img_path = pair[study][image]
        print(f"\t\t-Image: {img_path}")
        run(["clitkAffineTransform", "-i", img_path, "-o", img_path, "-l", like_img])


# Change the spacing of the images in the list.
def resample(image_paths, spacing):
    for img_path in image_paths:
        arg_list = ["clitkAffineTransform", "-i", img_path, "-o", img_path, "--adaptive",
                    f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}"]

        img = img_path.split("/")[-1]
        if "liver" in img or "tumor" in img:
            arg_list.append("--interp=0")

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

    mean = float(output_split[6][5:])
    std = 1 / float(output_split[9][6:])

    return -mean, std


# Perform a gaussian normalization to the images.
def gaussian_normalize(image_paths):
    for img_path in image_paths:
        stats = get_statistics(img_path)

        for idx, stat in enumerate(stats):
            run(["clitkImageArithm", "-i", img_path, "-o", img_path, "-s", str(stat), "-t", str(idx)])


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("parser/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)
    ct_study = args.std if args.std else "SPECT-CT"

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the respective output structures. Those will be used for registration.
    create_output_structures(input_dir, output_dir, identical=True)

    # Manually add the patients in this dictionary that have fragmented segmentations, in order to be added together
    # and have one segmentation file.
    sparse_seg = {
        "JohnDoe_ANON28177": {"tumor": ["ceMRI", "SPECT-CT"]},
        "JohnDoe_ANON39011": {"tumor": ["ceMRI"]},
        "JohnDoe_ANON55098": {"tumor": ["ceMRI", "SPECT-CT"]},
        "JohnDoe_ANON61677": {"tumor": ["ceMRI"]}
    }

    # Preprocessing for conventional registration usage.
    for patient in os.listdir(output_dir):
        patient_dir = os.path.join(output_dir, patient)
        studies_dir = [study for study in os.listdir(patient_dir) if study == "ceMRI" or study == ct_study]

        pair = get_pair_paths(patient_dir, studies_dir)
        fixed_images = [
            item for item in [
                pair[ct_study]["volume"],
                pair[ct_study]["liver"],
                pair[ct_study]["tumor"]
            ]
        ]

        if patient in sparse_seg:
            print(f"Adding segmentations for patient {patient}")
            add_segmentations(patient_dir, sparse_seg[patient])
        print(f"-Bias field correction for patient: {patient}")
        bias_field_correction(pair["MRI"]["volume"])
        print(f"-Set Spacing for patient: {patient}")
        # spacing = find_minimum_spacing(pair["CT"]["volume"], pair["MRI"]["volume"])
        resample(fixed_images, (1, 1, 1))
        # print(f"\n-Cropping for patient: {patient}")
        # crop(pair)
        print(f"-Resampling for patient: {patient}")
        resample_moving_2_fixed(pair, ct_study)
        print(f"-Create boundary boxes for patient: {patient}")
        create_bounding_boxes(pair)
        print()


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
