# Import necessary files and libraries.
import os
from subprocess import run
import SimpleITK as sITK
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


# Bring all the images of a patient into the same physical space of the chosen one.
# def resample(input_pairs, output_pairs):
#     ct_like = input_pairs["CT"]["volume"]
#     mri_like = input_pairs["MRI"]["volume"]
#
#     for study in input_pairs:
#         print(f"\t-Study: {study}")
#         for image in input_pairs[study]:
#             img_input = input_pairs[study][image]
#             img_output = output_pairs[study][image]
#             print(f"\t\t-Image: {img_input} ---> {img_output}")
#             run(["clitkAffineTransform", "-i", img_input, "-o", img_output, "-l",
#                  f"{ct_like}" if study == "MRI" else mri_like])


# For all the available study pairs, adjust to minimum or predefined spacing for each axis (in mm).
def change_spacing(input_pairs, output_pairs, spacing=None):
    # If there is not a predefined spacing, get the minimum spacing for each axis between the volumes
    if not spacing:
        spacing = find_minimum(input_pairs["CT"]["volume"], input_pairs["MRI"]["volume"])

    for study in input_pairs:
        print(f"\t-Study: {study}")
        for image in input_pairs[study]:
            img_input = input_pairs[study][image]
            img_output = output_pairs[study][image]
            print(f"\t\t-Image: {img_input} ---> {img_output}")

            run(["clitkAffineTransform", "-i", img_input, "-o", img_output, "--adaptive",
                 f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}"])


# Crop the pairs. For each study, crop automatically the mask and based on the mask, crop the volume.
def crop(input_pairs, output_pairs):
    for study in input_pairs:
        print(f"\t-Study: {study}")

        # Crop the liver mask automatically.
        liver_input = input_pairs[study]["liver"]
        liver_output = output_pairs[study]["liver"]

        print(f"\t\t-Nifty: {liver_input} ---> {liver_output}")
        run(["clitkAutoCrop", "-i", liver_input, "-o", liver_output])

        # Crop the volume and the tumor mask according to the cropped liver mask.
        volume_input = input_pairs[study]["volume"]
        volume_output = output_pairs[study]["volume"]
        tumor_input = input_pairs[study]["tumor"]
        tumor_output = output_pairs[study]["tumor"]

        print(f"\t\t-Nifty: {volume_input} ---> {volume_output}")
        run(["clitkCropImage", "-i", volume_input, "--like", liver_output, "-o", volume_output])
        print(f"\t\t-Nifty: {tumor_input} ---> {tumor_output}")
        run(["clitkCropImage", "-i", tumor_input, "--like", liver_output, "-o", tumor_output])


# TODO: Add the function from Felix' preprocessing file (the one in the USB key) that locates identical or very
#  similar patients.


def main():
    args = setup_parser("messages/preprocessing_parser.json")
    # Set required and optional arguments.
    input_dir, output_dir = args.i, args.o
    ct_study = args.s if args.s else "SPECT-CT"

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir, depth=2)

    # Do the required preprocessing for each of the patients.
    for patient in os.listdir(input_dir):
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        input_studies = [study for study in os.listdir(patient_input) if study == "ceMRI" or study == ct_study]
        output_studies = [study for study in os.listdir(patient_input) if study == "ceMRI" or study == ct_study]
        input_pair = get_pair_paths(patient_input, input_studies)
        output_pair = get_pair_paths(patient_output, output_studies)

        print(f"\n-Adjust Spacing for patient: {patient}")
        change_spacing(input_pair, output_pair)
        # print(f"\n-Affine Transform for patient: {patient}")
        # resample(input_pair, output_pair)
        print(f"\n-Cropping for patient: {patient}")
        crop(output_pair, output_pair)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
