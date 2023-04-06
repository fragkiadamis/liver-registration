# Import necessary files and libraries.
import os
import sys
from subprocess import check_output, CalledProcessError
from utils import setup_parser, validate_paths, \
    create_output_structures, create_dir, \
    update_dataframe, open_data_frame, dataframe_averages,\
    rename_instance, ConsoleColors


# Calculate the dice index between 2 RT structures
def calculate_dice(masks):
    liver_dice = check_output(["clitkDice", "-i", masks[0]['liver'], "-j", masks[1]['liver']])
    tumor_dice = check_output(["clitkDice", "-i", masks[0]['tumor'], "-j", masks[1]['tumor']])

    return {
        "liver": float(liver_dice.decode("utf-8")),
        "tumor": float(tumor_dice.decode("utf-8"))
    }


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
def apply_transform(transform_file, img_path, img_type, transformation_dir):
    # Change the bspline interpolation order from 3 to 0 to apply the transformation to the binary mask.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "3", "new": "0"})

    # Apply transformation to the mask.
    check_output(["transformix", "-in", img_path, "-out", transformation_dir, "-tp", transform_file])

    # Reset bspline interpolation order to 3.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "3", "new": "0"})

    # Rename the transformed file and return the path.
    rename_instance(transformation_dir, "result.nii.gz", f"{img_type}.nii.gz")


# Register the input pair using elastix and the defined parameters file.
def register_images(pair, img_type, output, par_file, props):
    # Create elastix argument list.
    elx_arguments = ["-f", pair["fixed"][img_type], "-m", pair["moving"][img_type], "-out", output, "-p", par_file]

    # Add initial transform file to the arguments, if available.
    if props["init_transform"]:
        elx_arguments.extend(["-t0", props["init_transform"]])

    # If the use of masks is defined, use them only to the files that are marked for mask usage.
    if props["masks"] and "_Mask" in par_file:
        elx_arguments.extend(["-fMask", pair["fixed"]["liver_mask"]])

    try:
        # Perform registration, rename the resulted image and return the image and the transformation.
        check_output(["elastix", *elx_arguments])
        result_img_path = rename_instance(output, "result.0.nii.gz", f"{img_type}.nii.gz")
        return result_img_path, os.path.join(output, "TransformParameters.0.txt")
    except CalledProcessError as e:
        print(f"\t\t{ConsoleColors.FAIL}Failed!{ConsoleColors.END}")
        return -1


# Return into a dictionary the fixed and moving paths
def get_pair_paths(patient_input, fixed_modality, moving_modality):
    return {
        "fixed": {
            "volume": os.path.join(patient_input, fixed_modality, "volume.nii.gz"),
            "liver": os.path.join(patient_input, fixed_modality, "liver.nii.gz"),
            "liver_bb": os.path.join(patient_input, fixed_modality, "liver_bb.nii.gz"),
            "tumor": os.path.join(patient_input, fixed_modality, "tumor.nii.gz")
        },
        "moving": {
            "volume": os.path.join(patient_input, moving_modality, "volume.nii.gz"),
            "liver": os.path.join(patient_input, moving_modality, "liver.nii.gz"),
            "liver_bb": os.path.join(patient_input, moving_modality, "liver_bb.nii.gz"),
            "tumor": os.path.join(patient_input, moving_modality, "tumor.nii.gz")
        }
    }


def main():
    args = setup_parser("messages/elastix_cli_parser.json")
    # Set required and optional arguments.
    input_dir, output_dir, parameters_dir = args.i, args.o, args.p
    fixed_modality = args.fm if args.fm else "SPECT-CT"
    moving_modality = args.mm if args.fm else "ceMRI"
    masks = True if args.masks else False
    img_type = args.t if args.t else "volume"

    # Avoid registering the masks with the use of the masks (see it in the future...)
    if masks and (img_type == "liver" or img_type == "tumor" or img_type == "liver_box"):
        sys.exit("Can't use masks while registering the masks...")

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    create_output_structures(input_dir, output_dir, depth=1)
    patients_list = os.listdir(input_dir)
    df = open_data_frame(patients_list)

    # Start registration for each patient in the dataset.
    print(f"\n{ConsoleColors.OK_GREEN}Registering {moving_modality} on {fixed_modality}{ConsoleColors.END}")
    for patient in patients_list:
        print(f"-Registering patient: {patient}")
        patient_input, patient_output = os.path.join(input_dir, patient), os.path.join(output_dir, patient)
        pair = get_pair_paths(patient_input, fixed_modality, moving_modality)

        # Calculate the initial dice index and save the results.
        dice = calculate_dice((pair["fixed"], pair["moving"]))
        for idx in dice:
            df = update_dataframe(df, patient, dice[idx], f"Initial {idx} dice")
        print(f"\t-Initial Indices\n\t\tLiver Dice: {dice['liver']} - Tumor Dice: {dice['tumor']}")

        registration = ""
        for par_file in os.listdir(parameters_dir):
            transform = par_file.split(".")[0]
            transform_output = create_dir(patient_output, transform)
            par_file = os.path.join(parameters_dir, par_file)

            # Start teh actual registration.
            print(f"\t-{transform} Transform...")
            props = {"init_transform": registration, "masks": masks}
            results_path, registration = register_images(pair, img_type, transform_output, par_file, props)

            # In case of failure to finish the registration process.
            if registration == -1:
                df = update_dataframe(df, patient, -1, f"{transform} Failed")
                break

            # Apply transformation of moving image to the other moving images and calculate dice.
            print(f"\t\t-Applying transform...")
            for img in pair["moving"]:
                # No need to apply the transform to the registered image.
                if img == img_type:
                    continue
                apply_transform(registration, pair["moving"][img], img, transform_output)

            # Calculate new dice on the transformed masks.
            transformed_pair = {
                "liver": f"{transform_output}/liver.nii.gz",
                "tumor": f"{transform_output}/tumor.nii.gz"
            }
            dice = calculate_dice((pair["fixed"], transformed_pair))
            for idx in dice:
                df = update_dataframe(df, patient, dice[idx], f"Initial {idx} dice")
            print(f"\t\t-New Indices\n\t\t\tLiver Dice: {dice['liver']} - Tumor Dice: {dice['tumor']}\n")

    dataframe_averages(df)
    df.to_excel("output.xlsx")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
