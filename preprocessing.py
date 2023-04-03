# Import necessary files and libraries.
import argparse
import os
from subprocess import run
import SimpleITK as sITK
from utils import validate_paths, create_output_structures


# Traverse through the given dataset paths and create paired paths between the available modalities.
def create_paired_paths(parent_dir, studies):
    pairs = {}
    for study in studies:
        pairs[f"{'CT' if 'CT' in study else 'MRI'}"] = {
            "volume": os.path.join(parent_dir, study, f"{study}_volume.nii.gz"),
            "rtstruct_liver": os.path.join(parent_dir, study, f"{study}_rtstruct_liver.nii.gz"),
            "rtstruct_tumor": os.path.join(parent_dir, study, f"{study}_rtstruct_tumor.nii.gz")
        }
    return pairs


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
def resample(input_pairs, output_pairs, spacing=None):
    # If there is not a predefined spacing, get the minimum spacing for each axis between the volumes
    if not spacing:
        spacing = find_minimum(input_pairs["CT"]["volume"], input_pairs["MRI"]["volume"], "spacing")

    for study in input_pairs:
        print(f"\t-Study: {study}")
        for image in input_pairs[study]:
            img_input = input_pairs[study][image]
            img_output = output_pairs[study][image]
            print(f"\t\t-Image: {img_input} ---> {img_output}")

            run(["clitkAffineTransform", "-i", img_input, "-o", img_output, "-l", input_pairs["CT"]["volume"]])

            # run(["clitkAffineTransform", "-i", img_path, "-o", img_path, "--adaptive",
            #               f"--spacing={str(spacing[0])},{str(spacing[1])},{str(spacing[2])}"])


# Crop the pairs. For each study, crop automatically the mask and based on the mask, crop the volume.
def crop(input_pairs, output_pairs):
    for study in input_pairs:
        print(f"\t-Study: {study}")

        # Crop the liver mask automatically.
        liver_input = input_pairs[study]["rtstruct_liver"]
        liver_output = output_pairs[study]["rtstruct_liver"]

        print(f"\t\t-Nifty: {liver_input} ---> {liver_output}")
        run(["clitkAutoCrop", "-i", liver_input, "-o", liver_output])

        # Crop the volume and the tumor mask according to the cropped liver mask.
        volume_input = input_pairs[study]["volume"]
        volume_output = output_pairs[study]["volume"]
        tumor_input = input_pairs[study]["rtstruct_tumor"]
        tumor_output = output_pairs[study]["rtstruct_tumor"]

        print(f"\t\t-Nifty: {volume_input} ---> {volume_output}")
        run(["clitkCropImage", "-i", volume_input, "--like", liver_output, "-o", volume_output])
        print(f"\t\t-Nifty: {tumor_input} ---> {tumor_output}")
        run(["clitkCropImage", "-i", tumor_input, "--like", liver_output, "-o", tumor_output])


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
    required_args.add_argument("-s", help="The CT study that is going to be processed e.g -c=SPECT-CT", required=True)
    args = parser.parse_args()

    input_dir, output_dir, ct_study = args.i, args.o, args.s

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir, depth=2)

    # Do the required preprocessing for each of the patients.
    for patient in os.listdir(input_dir):
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        input_studies = [study for study in os.listdir(patient_input) if study == "ceMRI" or study == ct_study]
        output_studies = [study for study in os.listdir(patient_input) if study == "ceMRI" or study == ct_study]
        input_pairs = create_paired_paths(patient_input, input_studies)
        output_pairs = create_paired_paths(patient_output, output_studies)

        print(f"\n-Resampling patient: {patient}")
        resample(input_pairs, output_pairs)
        print(f"\n-Cropping patient: {patient}")
        crop(output_pairs, output_pairs)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
