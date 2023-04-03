import argparse
import os
from subprocess import check_output, CalledProcessError

from utils import validate_paths, create_output_structures


# Calculate the dice index between 2 RT structures
def calculate_dice(items):
    dice_index = check_output(["clitkDice", "-i", items[0], "-j", items[1]])
    dice_index = float(dice_index.decode("utf-8"))

    return dice_index


# Open file and edit the bspline interpolation order.
def change_interpolation_order(file_path, order):
    # Read the transformation file and replace the line that refers to the interpolator order
    file = open(file_path, "r")
    replaced_content, replaced_line = "", ""
    for line in file:
        line = line.strip()
        if "FinalBSplineInterpolationOrder" in line:
            replaced_line = line.replace(order[0], order[1])
        else:
            replaced_line = line

        replaced_content = f"{replaced_content}{replaced_line}\n"
    file.close()

    # Overwrite the file with the new content
    file = open(file_path, "w")
    file.write(replaced_content)
    file.close()


# Apply the transformation parameters to the masks.
def apply_transform(transformation_file, patient_pair, transformation_dir):
    # Change the bspline interpolation order from 3 to 0 to apply the transformation to the binary mask. Then
    # apply the transformation on the mask and reset the interpolation order to the initial value.
    change_interpolation_order(transformation_file, ["3", "0"])
    check_output(["transformix", "-in", patient_pair[3], "-out", transformation_dir, "-tp", transformation_file])
    change_interpolation_order(transformation_file, ["0", "3"])

    # Return the resampled mask path.
    return os.path.join(transformation_dir, "result.nii.gz")


def register_pair(input_pair, transform, use_masks, output, par_file_path, t0=None):
    print(f"\t-{transform} Transform.")
    print(output)

    # Create elastix argument list.
    elx_arguments = ["-f", input_pair["fixed"]["volume"], "-m", input_pair["moving"]["volume"],
                     "-out", output, "-p", f"{par_file_path}"]

    # Add initial transform to the arguments, if available.
    if t0:
        elx_arguments.extend(["-t0", t0])

    # Add masks if the option is enabled.
    if use_masks:
        elx_arguments.extend(["-fMask", input_pair["fixed"]["mask"], "-mMask", input_pair["moving"]["mask"]])

    try:
        # Perform registration and replace the transformation file with the new one.
        check_output(["elastix", *elx_arguments])
    except CalledProcessError as e:
        print(f"\033[91m\tFailed!\033[0m")

    return ""


def main():
    parser = argparse.ArgumentParser(description="Register a pair of images using the elastix CLI.")
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-i", help="Path to the nifty files.", required=True)
    required_args.add_argument("-o", help="Path to the registration output", required=True)
    required_args.add_argument("-f", help="Specify fixed image e.g. -f SPECT-CT", required=True)
    required_args.add_argument("-m", help="Specify moving image e.g. -m ceMRI", required=True)
    required_args.add_argument("-p", help="Path to the parameter files", required=True)
    required_args.add_argument("-masks", help="Set to True if you use masks (default=False).")
    args = parser.parse_args()

    input_dir, output_dir, fixed = args.i, args.o, args.f
    moving, parameters_dir, use_masks = args.m, args.p, args.masks

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir, depth=1)

    # Start registration for each patient in the dataset.
    for patient in os.listdir(input_dir):
        print(f"\n-Registering patient: {patient}")
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        input_pairs = {
            "fixed": {
                "volume": os.path.join(patient_input, fixed, "volume.nii.gz"),
                "mask": os.path.join(patient_input, fixed, "rtstruct_liver.nii.gz")
            },
            "moving": {
                "volume": os.path.join(patient_input, moving, "volume.nii.gz"),
                "mask": os.path.join(patient_input, moving, "rtstruct_liver.nii.gz")
            }
        }

        init_transform = ""
        for par_file in os.listdir(parameters_dir):
            transform = par_file.split(".")[0]
            transform_output = os.path.join(patient_output, transform)
            os.mkdir(transform_output)
            par_file_path = os.path.join(parameters_dir, par_file)
            init_transform = register_pair(input_pairs, transform, use_masks, transform_output,
                                           par_file_path, t0=init_transform)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
