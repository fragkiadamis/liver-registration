import os
from shutil import copy

from preprocessing import get_median_spacing, resample, cast_to_type, gaussian_normalize, min_max_normalization
from utils import create_dir, setup_parser


# Create the structure which will be used to train the network for the automatic mask generation.
def create_training_structure(input_dir, training_set):
    for patient in os.listdir(input_dir):
        ct_study = os.path.join(input_dir, patient, "SPECT-CT")

        # Copy the labels and the images at the corresponding directories.
        for img in os.listdir(ct_study):
            img_path = os.path.join(ct_study, img)
            is_label = True if "liver" in img or "tumor" in img else False

            copy(
                img_path,
                os.path.join(training_set["labels" if is_label else "images"], f"{patient}_{img}.nii.gz")
            )

            # Make a copy of the unprocessed labels.
            if is_label:
                copy(img_path, os.path.join(training_set["unprocessed_labels"], f"{patient}_{img}.nii.gz"))


def create_training_structure_2(fixed_dir, moving_dir, training_set):
    patients = os.listdir(fixed_dir)
    for patient in patients:
        ct_volume = os.path.join(fixed_dir, patient, "ct_volume.nii.gz")
        ct_label = os.path.join(fixed_dir, patient, "ct_liver.nii.gz")
        mri_volume = os.path.join(moving_dir, patient, "01_Affine_KS", "mri_volume_reg.nii.gz")
        mri_label = os.path.join(moving_dir, patient, "01_Affine_KS", "mri_liver_reg.nii.gz")

        copy(ct_volume, os.path.join(training_set["images"], f"{patient}_fixed.nii.gz"))
        copy(ct_label, os.path.join(training_set["labels"], f"{patient}_fixed.nii.gz"))
        copy(mri_volume, os.path.join(training_set["images"], f"{patient}_moving.nii.gz"))
        copy(mri_label, os.path.join(training_set["labels"], f"{patient}_moving.nii.gz"))


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("config/preprocessing_parser.json")
    # input_dir = os.path.join(dir_name, args.i)
    fixed_dir, moving_dir = os.path.join(dir_name, args.i.split(",")[0]), os.path.join(dir_name, args.i.split(",")[1])
    output_dir = os.path.join(dir_name, args.o)

    # Create the input structure for deep learning.
    create_dir(dir_name, output_dir)
    training_set = {
        "images": create_dir(output_dir, "images"),
        "labels": create_dir(output_dir, "labels"),
        "unprocessed_labels": create_dir(output_dir, "non_resampled_labels")
    }

    # Create the output structure.
    # create_training_structure(input_dir, training_set)
    create_training_structure_2(fixed_dir, moving_dir, training_set)

    image_paths = [os.path.join(training_set["images"], path) for path in os.listdir(training_set["images"])]
    label_paths = [os.path.join(training_set["labels"], path) for path in os.listdir(training_set["labels"])]

    min_max_normalization(image_paths)

    # Preprocessing for deep learning usage.
    # print("Get median spacing...")
    # median_spacing = get_median_spacing(training_set["images"])
    # print("Resample images...")
    # resample(image_paths, median_spacing)
    # print("Resample labels...")
    # resample(label_paths, median_spacing)
    # print("Cast images to float...")
    # cast_to_type(image_paths, "float")
    # print("Image normalization...")
    # gaussian_normalize(image_paths)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
