# Import necessary files and libraries.
import argparse
import os
from subprocess import check_output
import SimpleITK as sITK
from utils import validate_paths, create_paired_paths, create_output_structures


# Find and return the minimum spacing for each axis between two volumes.
def find_minimum(path_1, path_2, prop):
    ct_volume = sITK.ReadImage(path_1)
    mri_volume = sITK.ReadImage(path_2)

    if prop == "spacing":
        ct_spacing, mri_spacing = ct_volume.GetSpacing(), mri_volume.GetSpacing()
        spacing = (
            min(ct_spacing[0], mri_spacing[0]),
            min(ct_spacing[1], mri_spacing[1]),
            min(ct_spacing[2], mri_spacing[2])
        )
        return spacing
    elif prop == "size":
        ct_size, mri_size = ct_volume.GetSize(), mri_volume.GetSize()
        size = (
            min(ct_size[0], mri_size[0]),
            min(ct_size[1], mri_size[1]),
            min(ct_size[2], mri_size[2])
        )
        return size


# For all the available study pairs, resample the images of the pair for all the patients.
def resample(patient, input_pairs, output_pairs, spacing=None):
    print(f"\n-Resampling patient: {patient}")

    # If there is not a predefined spacing, get the minimum spacing for each axis between the volumes
    if not spacing:
        spacing = find_minimum(input_pairs["CT"]["volume"], input_pairs["MRI"]["volume"], "spacing")

    for study in input_pairs:
        print(f"\t-Study: {study}")
        for image in input_pairs[study]:
            img_input = input_pairs[study][image]
            img_output = output_pairs[study][image]
            print(f"\t\t-Image: {img_input} ---> {img_output}")

            check_output(["clitkAffineTransform", "-i", img_input, "-o", img_output, "-l", input_pairs["CT"]["volume"]])

            # check_output(["clitkAffineTransform", "-i", img_path, "-o", img_path, "--adaptive",
            #               f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}"])


# Crop the pairs. For each study, crop automatically the mask and based on the mask, crop the volume.
def crop(patient, input_pairs, output_pairs):
    print(f"\n-Cropping patient: {patient}")

    for study in input_pairs:
        print(f"\t-Study: {study}")

        # Crop the mask automatically.
        liver_input_path = input_pairs[study]["rtstruct_liver"]
        liver_output_path = output_pairs[study]["rtstruct_liver"]

        print(f"\t\t-Image: {liver_input_path} ---> {liver_output_path}")
        check_output(["clitkAutoCrop", "-i", liver_input_path, "-o", liver_output_path])

        # Crop the volume and the tumor mask according to the cropped liver mask.
        volume_input_path = input_pairs[study]["volume"]
        volume_output_path = output_pairs[study]["volume"]
        tumor_input_path = input_pairs[study]["rtstruct_tumor"]
        tumor_output_path = output_pairs[study]["rtstruct_tumor"]

        print(f"\t\t-Image: {volume_input_path} ---> {volume_output_path}")
        check_output(["clitkCropImage", "-i", volume_input_path, "--like", liver_output_path, "-o", volume_output_path])
        print(f"\t\t-Image: {tumor_input_path} ---> {tumor_output_path}")
        check_output(["clitkCropImage", "-i", tumor_input_path, "--like", liver_output_path, "-o", tumor_output_path])


# TODO: Add the function from Felix' preprocessing file (the one in the USB key) that locates identical or very
#  similar patients.


def main():
    parser = argparse.ArgumentParser(description="Create a database where the processed nifty images are stored.")
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-i", help="path to the directory where is the nifty database", required=True)
    required_args.add_argument("-o", help="path to a directory where to save the processed database", required=True)
    # The user has to define which CT study is going to be processed for the later process of registration. If the
    # SPECT-CT, is going to be used later for the registration process along the ceMRI, then the user should define
    # -c SPECT-CT. -c PET-CT for the PET-CT/ceMRI registration respectively.
    required_args.add_argument("-c", help="The CT study that is going to be processed e.g -c=SPECT-CT", required=True)
    args = parser.parse_args()

    input_dir, output_dir, ct_study = args.i, args.o, args.c

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir)

    # Do the required preprocessing for each of the patients.
    for patient in os.listdir(input_dir):
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        input_studies, output_studies = os.listdir(patient_input), os.listdir(patient_output)

        input_studies_pair = [study for study in input_studies if study == ct_study or study == "ceMRI"]
        output_studies_pair = [study for study in input_studies if study == ct_study or study == "ceMRI"]

        input_pairs = create_paired_paths(patient_input, input_studies_pair)
        output_pairs = create_paired_paths(patient_output, output_studies_pair)

        resample(patient, input_pairs, output_pairs)
        crop(patient, output_pairs, output_pairs)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
