import os
from shutil import copy

from preprocessing import get_median_spacing, resample, cast_to_type, gaussian_normalize
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


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("parser/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)

    # Create the input structure for deep learning.
    create_dir(dir_name, output_dir)
    training_set = {
        "images": create_dir(output_dir, "images"),
        "labels": create_dir(output_dir, "labels"),
        "unprocessed_labels": create_dir(output_dir, "non_resampled_labels")
    }

    # Create the output structure.
    create_training_structure(input_dir, training_set)

    # Preprocessing for deep learning usage.
    print("Get median spacing...")
    median_spacing = get_median_spacing(training_set["images"])
    print("Resample images...")
    image_paths = [os.path.join(training_set["images"], path) for path in os.listdir(training_set["images"])]
    resample(image_paths, median_spacing)
    print("Resample labels...")
    label_paths = [os.path.join(training_set["labels"], path) for path in os.listdir(training_set["labels"])]
    resample(label_paths, median_spacing)
    print("Cast images to float...")
    cast_to_type(image_paths, "float")
    print("Image normalization...")
    gaussian_normalize(image_paths)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
