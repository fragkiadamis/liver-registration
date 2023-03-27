import os
from dotenv import load_dotenv
from utils import validate_paths, find_ct_mri_pairs, execute_shell_cmd, calculate_dice


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
    trx_arguments = ["-in", patient_pair[3], "-out", transformation_dir, "-tp", transformation_file]
    execute_shell_cmd("transformix", os.environ["ELX_BIN_PATH"], trx_arguments)
    change_interpolation_order(transformation_file, ["0", "3"])

    # Return the reampled mask path
    return os.path.join(transformation_dir, "result.nii.gz")


# TODO: Refactor this code.
# Register pair of images using the elastix cli.
def elx_cli_register(ct_mri_pairs, parameters_path, output_path):
    # Define here the transformations.
    transformations = ["rigid", "bspline"]
    transformation_file = ""
    resampled_pairs = {}

    counter = 0
    for patient in ct_mri_pairs:
        if counter == 2:
            break
        counter += 1

        # Create an output directory for the patient.
        patient_output_dir = os.path.join(output_path, patient)
        if not os.path.exists(patient_output_dir):
            os.mkdir(patient_output_dir)

        # Do all the transformations for the patient CT-MRI pair.
        for transformation in transformations:
            # Create an output directory for each transformation.
            transformation_dir = os.path.join(patient_output_dir, transformation)
            if not os.path.exists(transformation_dir):
                os.mkdir(transformation_dir)

            # Create elastix argument list.
            elx_arguments = ["-f", ct_mri_pairs[patient][0], "-m", ct_mri_pairs[patient][2],
                             "-out", transformation_dir, "-p", f"{parameters_path}/{transformation}.txt"]

            # Add initial transform to the arguments, if the patient has any available.
            if transformation_file:
                elx_arguments.extend(["-t0", transformation_file])

            try:
                # Perform registration and replace the transformation file with the new one.
                # print(f"{patient} CT-MRI pair {transformation} registration...")
                # execute_shell_cmd("elastix", os.environ["ELX_BIN_PATH"], elx_arguments)
                # print(f"Applying transformation to mask...")
                transformation_file = os.path.join(transformation_dir, "TransformParameters.0.txt")
                resampled_mask_path = apply_transform(transformation_file, ct_mri_pairs[patient], transformation_dir)
                resampled_pairs[patient] = [
                    ct_mri_pairs[patient][0],
                    ct_mri_pairs[patient][1],
                    ct_mri_pairs[patient][2],
                    resampled_mask_path
                ]

                # Calculate the dice index after each step.
                # calculate_dice(resampled_pairs)
            except NameError:
                print(f"Transformation {transformation} failed for {patient}: {NameError}")

    return resampled_pairs


def main():
    # Load environmental variables from .env file (create a .env file according to .env.example).
    load_dotenv()
    nifty_dir = os.path.abspath(os.environ["NIFTY_PATH"])
    parameters_path = os.path.abspath(os.environ["ELX_CLI_PARAM_PATH"])
    elx_cli_output = os.path.abspath(os.environ["ELX_CLI_OUTPUT_PATH"])
    validate_paths(nifty_dir, elx_cli_output)

    ct_mri_pairs = find_ct_mri_pairs(nifty_dir, studies=["SPECT-CT", "ceMRI"])
    elx_cli_register(ct_mri_pairs, parameters_path, elx_cli_output)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
