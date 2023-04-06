# Import necessary files and libraries.
import os
from shutil import copy
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


# Compare the two volumes two see if they are exact the same or similar.
def compare_volumes(path_a, path_b):
    volume_a, volume_b = nib.load(path_a).get_fdata(), nib.load(path_b).get_fdata()
    volume_a, volume_b = np.array(volume_a, dtype=np.int32), np.array(volume_b, dtype=np.int32)
    shape_a, shape_b = np.shape(volume_a), np.shape(volume_b)

    result = 2
    if shape_a == shape_b:
        volume_sub = volume_a - volume_b
        summation = np.sum(np.absolute(volume_sub))

        if summation == 0:
            result = -1
        elif summation < 10000:
            result = 0
        elif summation < 100000:
            result = 1

    return result


# Check for duplicate patients in the dataset. Exact duplicates will be removed automatically, very similar ones are
# going to be stored in the duplicates directory and will be handled manually by the user. handled manually by the user.
def check_for_duplicates(input_dir, patients):
    for patient_a in patients:
        print(f"-Checking for duplicates for {patient_a}")
        patient_a_path = os.path.join(input_dir, patient_a)
        for study in os.listdir(patient_a_path):
            print(f"\t-Checking study {study}")
            volume_a_path = os.path.join(patient_a_path, str(study), "volume.nii.gz")

            # Remove self and check on the rest of the patients.
            list_without_self = patients.copy()
            list_without_self.remove(patient_a)
            for patient_b in list_without_self:
                print(f"\t\t-Against patient {patient_b}")
                patient_b_path = os.path.join(input_dir, patient_b)
                volume_b_path = os.path.join(patient_b_path, str(study), "volume.nii.gz")

                print(f"\t\t\t-Comparing {volume_a_path} with {volume_b_path}")
                result = compare_volumes(volume_a_path, volume_b_path)
                if result == -1:
                    print("\t\t\t-These images are exactly the same")
                elif result == 0:
                    print("\t\t\t-These images might be the same patient")
                elif result == 1:
                    print("\t\t\t-These images look alike")
                elif result == 2:
                    print("\t\t\t-These images seem OK!")


# Perform a bias field correction for the MRIs.
def bias_field_correction(input_mris, output_mris):
    run(["clitkN4BiasFieldCorrection", "-i", input_mris["volume"], "-o", output_mris["volume"]])


# Bring all the images of a patient into the same physical space of the chosen one.
def resample(input_pairs, output_pairs, like_modality):
    # Get the image that gives the physical space reference.
    like_img = input_pairs[like_modality]["volume"]
    # Get the images of the study that are about to be resampled.
    study = "CT" if like_modality == "ceMRI" else "MRI"
    images = input_pairs[study]

    print(f"\t-Bring {study} in the physical space of {like_modality}")
    # Resample the images and store them in the respective output directory.
    for image in images:
        img_input, img_output = input_pairs[study][image], output_pairs[study][image]
        print(f"\t\t-Image: {img_input} ---> {img_output}")
        run(["clitkAffineTransform", "-i", img_input, "-o", img_output, "-l", like_img])

    # This copying mechanism is only needed if the resampling process is the first that takes place on the original
    # nifty data. In any other case, the reference modality images will be available in the processed directory so
    # this mechanism should be commented out.
    # print(f"\t-Copying {like_modality} images...")
    # for image in input_pairs[like_modality]:
    #     img_input, img_output = input_pairs[like_modality][image], output_pairs[like_modality][image]
    #     print(f"\t\t-Image: {img_input} ---> {img_output}")
    #     copy(img_input, img_output)


# For all the available study pairs, adjust to minimum or predefined spacing for each axis (in mm).
def change_spacing(input_pairs, output_pairs, spacing=None):
    # If there is not a predefined spacing, get the minimum spacing for each axis between the volumes
    if not spacing:
        spacing = find_minimum(input_pairs["CT"]["volume"], input_pairs["MRI"]["volume"])

    for study in input_pairs:
        print(f"\t-Study: {study}")
        for image in input_pairs[study]:
            img_input, img_output = input_pairs[study][image], output_pairs[study][image]
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


def main():
    args = setup_parser("messages/preprocessing_parser.json")
    # Set required and optional arguments.
    input_dir, output_dir = args.i, args.o
    ct_study = args.std if args.std else "SPECT-CT"
    rs_reference = args.rsr if args.rsr else "CT"

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir, depth=2)

    # Make a check to handle any possible duplicate data.
    # check_for_duplicates(input_dir, os.listdir(input_dir))

    # Do the required preprocessing for each of the patients.
    for patient in os.listdir(input_dir):
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        input_studies = [study for study in os.listdir(patient_input) if study == "ceMRI" or study == ct_study]
        output_studies = [study for study in os.listdir(patient_input) if study == "ceMRI" or study == ct_study]
        input_pair = get_pair_paths(patient_input, input_studies)
        output_pair = get_pair_paths(patient_output, output_studies)

        # print(f"-Bias field correction for patient: {patient}")
        # bias_field_correction(input_pair["MRI"], output_pair["MRI"])
        # print(f"\n-Adjust Spacing for patient: {patient}")
        # change_spacing(output_pair, output_pair)
        # print(f"\n-Resampling for patient: {patient}")
        # resample(output_pair, output_pair, rs_reference)
        # print(f"\n-Cropping for patient: {patient}")
        # crop(output_pair, output_pair)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
