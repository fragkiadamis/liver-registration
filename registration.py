# Import necessary files and libraries.
import json
import os
from subprocess import check_output, CalledProcessError

import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from utils import setup_parser, validate_paths, ConsoleColors, create_dir, rename_instance, delete_dir


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


# Calculate Dice, Mean Absolute Distance etc. Use comments to include/exclude metrics.
def calculate_metrics(ground_truth, moving):
    ground_truth = nib.load(ground_truth)
    moving = nib.load(moving)

    # For overlap metrics
    ground_truth_data = np.array(ground_truth.get_fdata()).astype(int)
    moving_data = np.array(moving.get_fdata()).astype(int)
    ground_truth_sum = np.sum(ground_truth_data)
    moving_sum = np.sum(moving_data)
    intersection = ground_truth_data & moving_data
    # union = ground_truth_data | moving_data
    intersection_sum = np.count_nonzero(intersection)
    # union_sum = np.count_nonzero(union)

    # For distance metrics
    ground_truth_coords = np.array(np.where(ground_truth_data == 1)).T
    moving_coords = np.array(np.where(moving_data == 1)).T

    # Create the distance matrix with samples because the "on" values on both images are so many that the program
    # overflows the memory while trying to create the distance matrix.
    # n_samples = 10000
    # ground_truth_sample = np.random.choice(ground_truth_coords.shape[0], n_samples, replace=True)
    # moving_sample = np.random.choice(moving_coords.shape[0], n_samples, replace=True)

    # Calculate the distance matrix between the sampled "on" voxels in each image
    # dist_matrix = cdist(ground_truth_coords[ground_truth_sample], moving_coords[moving_sample])

    # Calculate the directed Hausdorff distance between the images
    distance_gt_2_moving = directed_hausdorff(ground_truth_coords, moving_coords)[0]
    distance_moving_2_gt = directed_hausdorff(moving_coords, ground_truth_coords)[0]

    return {
        "Dice": 2 * intersection_sum / (ground_truth_sum + moving_sum),
        # "Jaccard": intersection_sum / union_sum,
        # "M.A.D": np.mean(np.abs(dist_matrix)),
        "H.D": max(distance_gt_2_moving, distance_moving_2_gt)
    }


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/elastix_cli_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    pipeline_file = os.path.join(dir_name, args.pl)
    patient = args.p
    patient_input = os.path.join(input_dir, patient)
    output_dir = create_dir(patient_input, args.o)

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)

    # Open file and create the pipeline dictionary from json.
    pl = open(pipeline_file)
    pipeline = json.loads(pl.read())
    pl.close()

    print(f"{ConsoleColors.HEADER}Pipeline: {pipeline['name']}{ConsoleColors.END}")
    print(f"-Registering patient: {ConsoleColors.OK_BLUE}{patient}.{ConsoleColors.END}")

    results, evaluation_masks = {}, {}
    evaluation_masks.update({
        img: {
            "fixed": os.path.join(patient_input, f"{pipeline['studies']['fixed']}_{img}.nii.gz"),
            "moving": os.path.join(patient_input, f"{pipeline['studies']['moving']}_{img}.nii.gz")
        } for img in pipeline["evaluate"]
    })

    print(f"\t-Calculating Metrics.")
    for mask in evaluation_masks:
        results[mask] = {
            "Initial": calculate_metrics(evaluation_masks[mask]["fixed"], evaluation_masks[mask]["moving"])
        }
        print(f"\t\t-{mask}: {results[mask]}.")

    # Delete pipeline directory if it already exists and create a new one.
    pipeline_output = create_dir(output_dir, pipeline["name"])

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
        transform_output = create_dir(pipeline_output, transform_name)

        # Perform the registration, which will return a transformation file. This transformation will be used in
        # the next registration step of the pipeline as initial transform "t0".
        print(f"\t-Transform: {transform_name} on {step['images']['moving']}.")
        registration = elastix_cli(images, masks, parameters_file, transform_output, step["result"], registration)

        # In case of failure to finish the registration process, continue to the next patient.
        if registration == -1:
            break

        # If it's defined in the pipeline, apply the transform to the volume too. This is necessary for the cases
        # where all the registrations are done only between masks and no volumes.

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

    with open(f"{pipeline_output}/evaluation.json", "w") as fp:
        json.dump(results, fp)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
