# Import necessary files and libraries.
import os
import numpy as np
from subprocess import run
import SimpleITK as sITK
import nibabel as nib
from utils import setup_parser, validate_paths, create_output_structures


# Traverse through the given dataset paths and create paired paths between the available modalities.
def get_pair_paths(parent_dir, studies):
    pair = {}
    for study in studies:
        pair[f"{'CT' if 'CT' in study else 'MRI'}"] = {
            "volume": os.path.join(parent_dir, study, "volume.nii.gz"),
            "liver": os.path.join(parent_dir, study, "liver.nii.gz"),
            "tumor": os.path.join(parent_dir, study, "tumor.nii.gz")
        }
    return pair


# Find and return the minimum spacing for each axis between two volumes.
def find_minimum(path_1, path_2):
    ct_volume = sITK.ReadImage(path_1)
    mri_volume = sITK.ReadImage(path_2)

    ct_spacing, mri_spacing = ct_volume.GetSpacing(), mri_volume.GetSpacing()
    spacing = (
        min(ct_spacing[0], mri_spacing[0]),
        min(ct_spacing[1], mri_spacing[1]),
        min(ct_spacing[2], mri_spacing[2])
    )
    return spacing


# Perform a bias field correction for the MRIs.
def bias_field_correction(mri):
    run(["clitkN4BiasFieldCorrection", "-i", mri, "-o", mri])


# Bring all the images of a patient into the same physical space of the chosen one.
def resample(pair, like_modality):
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


# For all the available study pairs, adjust to minimum or predefined spacing for each axis (in mm).
def change_spacing(pair, spacing=None):
    # If there is not a predefined spacing, get the minimum spacing for each axis between the volumes
    if not spacing:
        spacing = find_minimum(pair["CT"]["volume"], pair["MRI"]["volume"])

    for study in pair:
        print(f"\t-Study: {study}")
        for image in pair[study]:
            img_path = pair[study][image]
            print(f"\t\t-Image: {img_path}")

            run(["clitkAffineTransform", "-i", img_path, "-o", img_path, "--adaptive",
                 f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}"])


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


def main():
    args = setup_parser("parser/preprocessing_parser.json")
    # Set required and optional arguments.
    input_dir, output_dir = args.i, args.o
    ct_study = args.std if args.std else "SPECT-CT"
    rs_reference = args.rsr if args.rsr else "CT"

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir, identical=True)

    # Do the required preprocessing for each of the patients.
    for patient in os.listdir(output_dir):
        patient_dir = os.path.join(output_dir, patient)
        studies_dir = [study for study in os.listdir(patient_dir) if study == "ceMRI" or study == ct_study]

        pair = get_pair_paths(patient_dir, studies_dir)

        # print(f"\n-Bias field correction for patient: {patient}")
        # bias_field_correction(pair["MRI"]["volume"])
        # print(f"\n-Adjust Spacing for patient: {patient}")
        # change_spacing(pair)
        # print(f"\n-Resampling for patient: {patient}")
        # resample(pair, rs_reference)
        # print(f"\n-Cropping for patient: {patient}")
        # crop(pair)
        print(f"\n-Create boundary boxes for patient: {patient}")
        create_bounding_boxes(pair)
        break


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
