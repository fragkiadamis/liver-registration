# Import necessary files and libraries.
import json
import os
from subprocess import check_output, CalledProcessError

from utils import setup_parser, validate_paths, ConsoleColors, create_dir, rename_instance, calculate_metrics


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
    for img_path in images:
        if "volume" not in img_path:
            # Change the bspline interpolation order from 3 to 0 to apply the transformation to the binary mask.
            change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "3", "new": "0"})

        # Set the transformix arguments.
        trx_args = ["-in", img_path, "-out", transformation_dir, "-tp", transform_file]
        if def_field:
            trx_args.extend(["-def", "all"])

        # Apply transformation.
        check_output(["transformix", *trx_args])
        filename = img_path.split("/")[-1].split(".")[0]
        rename_instance(transformation_dir, "result.nii.gz", f"{filename}_reg.nii.gz")

    # Reset bspline interpolation order to 3. If it's not changed, nothing is going to happen.
    change_transform_file(transform_file, "FinalBSplineInterpolationOrder", {"old": "0", "new": "3"})


# Register the input pair using elastix and the defined parameters file.
def elastix_cli(images, masks, parameters, output, filename, t0=None):
    # Create elastix argument list.
    elx_args = ["-f", images["fixed"], "-m", images["moving"], "-out", output, "-p", parameters, "-priority", "high"]

    # Add initial transform file to the arguments, if available.
    if t0:
        elx_args.extend(["-t0", t0])

    # Use the defined masks.
    if masks:
        if masks["fixed"]:
            elx_args.extend(["-fMask", masks["fixed"]])
        if masks["moving"]:
            elx_args.extend(["-mMask", masks["moving"]])

    try:
        # Perform registration, rename the resulted image and return the transformation.
        check_output(["elastix", *elx_args])
        rename_instance(output, "result.0.nii.gz", f"{filename}.nii.gz")
        return os.path.join(output, "TransformParameters.0.txt")
    except CalledProcessError as e:
        # Catch possible error
        print(f"\t\t{ConsoleColors.FAIL}Failed!{ConsoleColors.END}")
        return -1


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/elastix_cli_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    pipeline_file = os.path.join(dir_name, args.pl)
    patient = args.p
    patient_input = os.path.join(input_dir, patient)
    output_dir = create_dir(dir_name, args.o)

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)

    # Open file and create the pipeline dictionary from json.
    pl = open(pipeline_file)
    pipeline = json.loads(pl.read())
    pl.close()

    print(f"{ConsoleColors.HEADER}Pipeline: {pipeline['name']}{ConsoleColors.END}")
    print(f"-Registering patient: {ConsoleColors.OK_BLUE}{patient}.{ConsoleColors.END}")

    results, evaluation_masks = {}, {}
    for img in pipeline["evaluate"]:
        fixed_path = os.path.join(patient_input, f"{pipeline['studies']['fixed']}_{img}.nii.gz")
        moving_path = os.path.join(patient_input, f"{pipeline['studies']['moving']}_{img}.nii.gz")
        if os.path.exists(fixed_path) and os.path.exists(moving_path):
            evaluation_masks.update({
                img: {
                    "fixed": fixed_path,
                    "moving": moving_path
                }
            })

    print(f"\t-Calculating Metrics.")
    for mask in evaluation_masks:
        results[mask] = {
            "Initial": calculate_metrics(evaluation_masks[mask]["fixed"], evaluation_masks[mask]["moving"])
        }
        print(f"\t\t-{mask}: {results[mask]}.")

    # Delete pipeline directory if it already exists and create a new one.
    pipeline_output = create_dir(output_dir, pipeline["name"])
    patient_out = create_dir(pipeline_output, patient)

    registration = None
    for step in pipeline["registration_steps"]:
        # Get transform's properties.
        images, masks = {
            "fixed": os.path.join(patient_input, step["images"]["fixed"]),
            "moving": os.path.join(patient_input, step["images"]["moving"])
        }, {
            "fixed": os.path.join(patient_input, step["masks"]["fixed"]),
            "moving": os.path.join(patient_input, step["masks"]["moving"])
        } if "masks" in step else None

        parameters_file, transform_name = os.path.join(dir_name, step["parameters"]), step["name"]
        transform_output = create_dir(patient_out, transform_name)

        # Perform the registration, which will return a transformation file. This transformation will be used in
        # the next registration step of the pipeline as initial transform "t0".
        print(f"\t-Transform: {transform_name} on {step['images']['moving']}.")
        registration = elastix_cli(images, masks, parameters_file, transform_output, step["result"], registration)

        # In case of failure to finish the registration process, continue to the next patient.
        if registration == -1:
            break

        # Apply the transformation on the moving images.
        print(f"\t-Apply transform.")
        apply_on = [os.path.join(patient_input, x) for x in step["apply_on"]]
        apply_transform(registration, apply_on, transform_output, step["def_field"])
        for mask in evaluation_masks:
            evaluation_masks[mask]["moving"] = os.path.join(transform_output, f"mri_{mask}_reg.nii.gz")

        print(f"\t-Calculating Metrics.")
        for mask in evaluation_masks:
            results[mask].update({
                transform_name: calculate_metrics(evaluation_masks[mask]["fixed"], evaluation_masks[mask]["moving"])
            })
            print(f"\t\t-{mask}: {results[mask]}.")

    with open(f"{patient_out}/evaluation.json", "w") as fp:
        json.dump(results, fp)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
