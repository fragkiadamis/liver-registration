import argparse
import os
from subprocess import run, check_output, CalledProcessError
from utils import validate_paths, create_output_structures, create_dir


# Calculate the dice index between 2 RT structures
def calculate_dice(masks):
    dice_index = check_output(["clitkDice", "-i", masks[0], "-j", masks[1]])
    dice_index = float(dice_index.decode("utf-8"))

    return dice_index


# Open file and edit the bspline interpolation order.
def change_transform_file(file_path, prop, values):
    # Read the transformation file and replace the line that refers to the interpolator order
    file = open(file_path, "r")
    replaced_content, replaced_line = "", ""
    for line in file:
        line = line.strip()
        if prop in line:
            replaced_line = line.replace(values["old"], values["new"])
        else:
            replaced_line = line

        replaced_content = f"{replaced_content}{replaced_line}\n"
    file.close()

    # Overwrite the file with the new content
    file = open(file_path, "w")
    file.write(replaced_content)
    file.close()


# Apply the transformation parameters to the masks.
def apply_transform(transform_file, mask, transformation_dir):
    # Change the bspline interpolation order from 3 to 0 to apply the transformation to the binary mask.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "3", "new": "0"})

    # Apply transformation to the mask.
    run(["transformix", "-in", mask, "-out", transformation_dir, "-tp", transform_file])

    # Reset bspline interpolation order to 3.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "3", "new": "0"})

    # Return the resampled mask path.
    return os.path.join(transformation_dir, "result.nii.gz")


# Register the input pair using elastix and the defined parameters file.
def register_volumes(pair, output, par_file, props):
    # Create elastix argument list.
    elx_arguments = ["-f", pair["fixed"]["volume"], "-m", pair["moving"]["volume"],
                     "-out", output, "-p", f"{par_file}"]

    # Add initial transform file to the arguments, if available.
    if props["t0"]:
        elx_arguments.extend(["-t0", props["t0"]])

    # Add masks if the option is enabled.
    if props["masks"]:
        elx_arguments.extend(["-fMask", pair["fixed"]["liver_mask"], "-mMask", pair["moving"]["liver_mask"]])

    transform_file = ""
    try:
        # Perform registration and replace the transformation file with the new one.
        run(["elastix", *elx_arguments])
        transform_file = os.path.join(output, "TransformParameters.0.txt")
        print(f"\t\t-Applying transform to moving mask.")
        transformed_liver_mask = apply_transform(transform_file, pair["moving"]["liver_mask"], output)
        dice_index = calculate_dice((pair["fixed"]["liver_mask"], transformed_liver_mask))
        print(f"\t\t-Dice Index {dice_index}.")
    except CalledProcessError as e:
        print(f"\t\033[91m\tFailed!\033[0m")

    # Return the transform file so that it can be used to the next transformation.
    return transform_file


def main():
    parser = argparse.ArgumentParser(description="Register a pair of images using the elastix CLI.")
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-i", help="Path to the nifty files.", required=True)
    required_args.add_argument("-o", help="Path to the registration output", required=True)
    required_args.add_argument("-fm", help="Specify fixed modality e.g. -fm SPECT-CT", required=True)
    required_args.add_argument("-mm", help="Specify moving modality e.g. -mm ceMRI", required=True)
    required_args.add_argument("-p", help="Path to the parameter files", required=True)
    required_args.add_argument("-masks", help="Set to True if you use masks (default=False).")
    args = parser.parse_args()

    input_dir, output_dir, fixed_modality = args.i, args.o, args.fm
    moving_modality, parameters_dir, masks = args.mm, args.p, args.masks

    # Validate input and output paths.
    validate_paths(input_dir, output_dir)

    # Create the output respective structures.
    create_output_structures(input_dir, output_dir, depth=1)

    # Start registration for each patient in the dataset.
    for patient in os.listdir(input_dir):
        print(f"\n-Registering patient: {patient}")
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        pair = {
            "fixed": {
                "volume": os.path.join(patient_input, fixed_modality, f"{fixed_modality}_volume.nii.gz"),
                "liver_mask": os.path.join(patient_input, fixed_modality, f"{fixed_modality}_rtstruct_liver.nii.gz")
            },
            "moving": {
                "volume": os.path.join(patient_input, moving_modality, f"{moving_modality}_volume.nii.gz"),
                "liver_mask": os.path.join(patient_input, moving_modality, f"{moving_modality}_rtstruct_liver.nii.gz")
            }
        }

        init_transform = ""
        for par_file in os.listdir(parameters_dir):
            transform = par_file.split(".")[0]
            transform_output = create_dir(patient_output, transform)
            par_file = os.path.join(parameters_dir, par_file)
            print(f"\t-{transform} Transform.")
            init_transform = register_volumes(pair, transform_output, par_file, {"t0": init_transform, "masks": masks})
        break


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
