# Import necessary files and libraries.
import copy
import json
import os
from subprocess import check_output, CalledProcessError
from utils import setup_parser, validate_paths, \
    create_output_structures, create_dir, \
    update_dataframe, open_data_frame, dataframe_averages,\
    rename_instance, ConsoleColors


def get_mask_paths(patient, studies, masks):
    mask_pair = {}
    for mask in masks:
        mask_pair[mask] = {
            "fixed": os.path.join(patient, studies["fixed"], mask),
            "moving": os.path.join(patient, studies["moving"], mask),
        }

    return mask_pair


# Return into a dictionary the fixed and moving paths
def get_image_paths(patient_input, studies, images):
    return {
        "fixed": os.path.join(patient_input, studies["fixed"], images["fixed"]),
        "moving": os.path.join(patient_input, studies["moving"], images["moving"])
    }


# Calculate the dice index between 2 RT structures
def calculate_dice(masks):
    dice_indices = {}
    for mask in masks:
        dice = check_output(["clitkDice", "-i", masks[mask]['fixed'], "-j", masks[mask]['moving']])
        dice_indices[mask] = float(dice.decode("utf-8"))

    return dice_indices


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
def apply_transform(transform_file, evaluation_masks, transformation_dir):
    # Change the bspline interpolation order from 3 to 0 to apply the transformation to the binary mask.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "3", "new": "0"})

    masks = copy.deepcopy(evaluation_masks)
    for mask in masks:
        # Apply transformation to the mask.
        check_output(["transformix", "-in", masks[mask]["moving"], "-out", transformation_dir, "-tp", transform_file])
        masks[mask]["moving"] = rename_instance(transformation_dir, "result.nii.gz", f"{mask}_result.nii.gz")

    # Reset bspline interpolation order to 3.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "0", "new": "3"})

    return masks


# Register the input pair using elastix and the defined parameters file.
def elastix_cli(images, parameters, output, masks=None, t0=None):
    # Create elastix argument list.
    elx_arguments = ["-f", images["fixed"], "-m", images["moving"], "-out", output, "-p", parameters]

    # Add initial transform file to the arguments, if available.
    if t0:
        elx_arguments.extend(["-t0", t0])

    # Use masks if it is defined.
    if masks:
        if masks["fixed"]:
            elx_arguments.extend(["-fMask", masks["fixed"]])
        if masks["moving"]:
            elx_arguments.extend(["-mMask", masks["moving"]])

    try:
        # Perform registration, rename the resulted image and return the transformation.
        check_output(["elastix", *elx_arguments])
        return os.path.join(output, "TransformParameters.0.txt")
    except CalledProcessError as e:
        # Catch possible error
        print(f"\t\t{ConsoleColors.FAIL}Failed!{ConsoleColors.END}")
        return -1


def main():
    args = setup_parser("parser/elastix_cli_parser.json")
    # Set required and optional arguments.
    input_dir, output_dir, pipeline_file = args.i, args.o, args.p

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    create_output_structures(input_dir, output_dir, depth=1)

    if not os.path.exists("results"):
        os.mkdir("results")
    patients_list = os.listdir(input_dir)

    with open(pipeline_file, 'r') as pl:
        pipeline = json.load(pl)

    df = open_data_frame(patients_list)
    print(f"{ConsoleColors.HEADER}Pipeline: {pipeline['name']}{ConsoleColors.END}")

    # Start registration for each patient in the dataset.
    for patient in patients_list:
        print(f"-Registering patient: {ConsoleColors.OK_BLUE}{patient}{ConsoleColors.END}")
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        evaluation_masks = get_mask_paths(patient_input, pipeline["studies"], pipeline["evaluate_on"])

        # Calculate the initial dice index and save the results.
        dice = calculate_dice(evaluation_masks)
        print("\t-Initial dice index.")
        for idx in dice:
            df = update_dataframe(df, patient, dice[idx], f"Initial {idx} dice")
            print(f"\t\t-{idx}: {ConsoleColors.UNDERLINE}{dice[idx]}.{ConsoleColors.END}")

        # Create the output of the pipeline and execute its steps.
        pipeline_output = create_dir(patient_output, pipeline["name"])
        registration = None
        for step in pipeline["registration_steps"]:
            images = get_image_paths(patient_input, pipeline["studies"], step["images"])
            parameters_file = step["parameters"]
            masks, transform_name = step["masks"], step["name"]
            transform_output = create_dir(pipeline_output, transform_name)

            # Perform the registration, which will return a transformation file. This transformation will be used in
            # the next registration step of the pipeline as initial transform "t0".
            print(f"\t-Transform: {transform_name} on {step['images']['moving']}")
            registration = elastix_cli(images, parameters_file, transform_output, masks, registration)

            # In case of failure to finish the registration process.
            if registration == -1:
                df = update_dataframe(df, patient, -1, f"{transform_name} Failed")
                break

            # Apply the transformation on the moving masks.
            print(f"\t-Apply transform to masks.")
            transformed_masks = apply_transform(registration, evaluation_masks, transform_output)

            # Recalculate dice index.
            dice = calculate_dice(transformed_masks)
            print("\t-New dice index.")
            for idx in dice:
                df = update_dataframe(df, patient, dice[idx], f"{transform_name} {idx} dice")
                print(f"\t\t-{idx}: {ConsoleColors.UNDERLINE}{dice[idx]}.{ConsoleColors.END}")

            print()

    dataframe_averages(df)
    df.to_excel(f"results/{pipeline['name']}.xlsx")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
