# Import necessary files and libraries.
from copy import deepcopy
import json
import os
from subprocess import check_output, CalledProcessError
from utils import setup_parser, validate_paths, \
    create_output_structures, create_dir, \
    update_dataframe, open_data_frame, dataframe_stats, \
    rename_instance, ConsoleColors, save_dfs, delete_dir


def get_mask_paths(patient, studies, masks):
    mask_pair = {}
    for mask in masks:
        mask_pair[mask] = {
            "fixed": os.path.join(patient, studies["fixed"], mask, ".nii.gz"),
            "moving": os.path.join(patient, studies["moving"], mask, ".nii.gz")
        }

    return mask_pair


# Return into a dictionary the fixed and moving paths
def get_image_paths(patient_input, studies, images):
    return {
        "fixed": os.path.join(patient_input, studies["fixed"], images["fixed"], ".nii.gz"),
        "moving": os.path.join(patient_input, studies["moving"], images["moving"], ".nii.gz")
        if "moving" in images else ""
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
def apply_transform(transform_file, images, transformation_dir, def_field):
    for img in images:
        if img != "volume":
            # Change the bspline interpolation order from 3 to 0 to apply the transformation to the binary mask.
            change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "3", "new": "0"})

        # Set the transformix arguments.
        trx_args = ["-in", images[img]["moving"], "-out", transformation_dir, "-tp", transform_file]
        if def_field:
            trx_args.extend(["-def", "all"])

        # Apply transformation.
        check_output(["transformix", *trx_args])
        images[img]["moving"] = rename_instance(transformation_dir, "result.nii.gz", f"{img}_result.nii.gz")

    # Reset bspline interpolation order to 3. If it's not changed, nothing is going to happen.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "0", "new": "3"})

    return images


# Register the input pair using elastix and the defined parameters file.
def elastix_cli(images, masks, parameters, output, t0=None):
    # Create elastix argument list.
    elx_arguments = ["-f", images["fixed"], "-m", images["moving"], "-out", output, "-p", parameters]

    # Add initial transform file to the arguments, if available.
    if t0:
        elx_arguments.extend(["-t0", t0])

    # Use the defined masks.
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
    input_dir, output_dir, pipeline_file = args.i, args.o, args.p

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    if not os.path.exists(output_dir):
        create_output_structures(input_dir, output_dir, depth=1)

    # Open file and create the pipeline dictionary from json.
    pl = open(pipeline_file)
    pipeline = json.loads(pl.read())
    pl.close()

    print(f"{ConsoleColors.HEADER}Pipeline: {pipeline['name']}{ConsoleColors.END}")

    # Create the results dataframe.
    patients_list = os.listdir(input_dir)
    results_path = f"{create_dir('.', 'results')}/{pipeline['name']}.xlsx"
    dfs = open_data_frame(patients_list, pipeline["evaluate_on"], results_path)

    # Start registration for each patient in the dataset.
    for patient in patients_list:
        print(f"-Registering patient: {ConsoleColors.OK_BLUE}{patient}.{ConsoleColors.END}")
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        evaluation_masks = get_mask_paths(patient_input, pipeline["studies"], pipeline["evaluate_on"])

        # Calculate the initial dice index and save the results.
        dice = calculate_dice(evaluation_masks)
        dfs = update_dataframe(dfs, patient, dice, "Initial dice", results_path)

        # Delete pipeline directory if it already exists and create a new one.
        pipeline_output = os.path.join(patient_output, pipeline["name"])
        delete_dir(pipeline_output)
        create_dir(patient_output, pipeline["name"])

        registration = None
        for step in pipeline["registration_steps"]:
            # Get transform's properties.
            images = get_image_paths(patient_input, pipeline["studies"], step["images"])
            masks = get_image_paths(patient_input, pipeline["studies"], step["masks"]) if "masks" in step else None

            parameters_file, transform_name = step["parameters"], step["name"]

            # Create current transform's output.
            transform_output = create_dir(pipeline_output, transform_name)

            # Perform the registration, which will return a transformation file. This transformation will be used in
            # the next registration step of the pipeline as initial transform "t0".
            print(f"\t-Transform: {transform_name} on {step['images']['moving']}.")
            registration = elastix_cli(images, masks, parameters_file, transform_output, registration)

            # In case of failure to finish the registration process, continue to the next patient.
            if registration == -1:
                break

            # If it's defined in the pipeline, apply the transform to the volume too. This is necessary for the cases
            # where all the registrations are done only between masks and no volumes.
            image_set = deepcopy(evaluation_masks)
            if pipeline["apply_on_volume"]:
                image_set["volume"] = {
                    "fixed": os.path.join(patient_input, pipeline["studies"]["fixed"], "volume.nii.gz"),
                    "moving": os.path.join(patient_input, pipeline["studies"]["moving"], "volume.nii.gz")
                }

            # Apply the transformation on the moving images.
            print(f"\t-Apply transform to masks.")
            transformed = apply_transform(registration, image_set, transform_output, step["def_field"])

            # Recalculate dice index only to transformed masks.
            transformed_masks = {x: transformed[x] for x in transformed if x != "volume"}
            dice = calculate_dice(transformed_masks)
            dfs = update_dataframe(dfs, patient, dice, f"{transform_name}", results_path)
        print()

    # Get statistics for the dataframes and save.
    dataframe_stats(dfs)
    save_dfs(dfs, results_path)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
