import os
import fnmatch
from dotenv import load_dotenv
from utils import execute_shell_cmd


# Traverse through the data and find the CT and MRI structs for the respective anatomy
def find_respective_structs(root_dir, anatomy):
    rt_structs = {}

    for patient in os.listdir(root_dir):
        dir_spect_ct = os.path.join(root_dir, patient, "SPECT-CT")
        dir_mri = os.path.join(root_dir, patient, "ceMRI")

        # Find and get the liver structure for the MRI and the CT
        mri_struct, ct_liver_struct = "", ""
        for nii_file in os.listdir(dir_mri):
            if fnmatch.fnmatch(nii_file, f"*{anatomy}*.nii.gz"):
                mri_struct = os.path.join(dir_mri, nii_file)

        for nii_file in os.listdir(dir_spect_ct):
            if fnmatch.fnmatch(nii_file, f"*{anatomy}*.nii.gz"):
                ct_liver_struct = os.path.join(dir_spect_ct, nii_file)

        rt_structs[f"{patient}"] = [ct_liver_struct, mri_struct]

    return rt_structs


# Resample MRI RT structures in the physical space of CT RT structures. To calculate initial dice we need to
# the two objects to be in the same physical space
def resample_structs(rt_structs):
    for patient in rt_structs:
        print(f"{patient} MRI struct resampling")
        ct_struct = rt_structs[f"{patient}"][0]
        mri_struct = rt_structs[f"{patient}"][1]

        split_path = mri_struct.split(".")
        resampled_mri_struct = f"{split_path[0]}_resampled.{split_path[1]}.{split_path[2]}"
        execute_shell_cmd("clitkAffineTransform", ["-i", mri_struct, "-o", resampled_mri_struct, "-l", ct_struct])

        rt_structs[f"{patient}"][1] = resampled_mri_struct

    print("Resampling Done.")
    return rt_structs


# Calculate the dice index between 2 RT structures
def calculate_dice(rt_structs):
    dice_results = {}

    for patient in rt_structs:
        ct_struct = rt_structs[f"{patient}"][0]
        mri_struct = rt_structs[f"{patient}"][1]
        dice_index = execute_shell_cmd("clitkDice", ["-i", ct_struct, "-j", mri_struct])
        dice_index = float(dice_index.decode("utf-8"))

        print(f"{patient} Dice Index: {dice_index}")
        dice_results[f"{patient}"] = dice_index

    return dice_results


# Delete the resampled RT structures.
def delete_resampled_structs(rt_structs):
    for patient in rt_structs:
        mri_struct = rt_structs[f"{patient}"][1]
        os.remove(mri_struct)


def main():
    # Load environmental variables from .env file (create a .env file according to .env.example).
    load_dotenv()

    # Parse CLI arguments and handle inputs and outputs.
    nifty_dir = os.environ["NIFTY_PATH"]

    # Calculate initial overlap between.
    rt_structs = find_respective_structs(nifty_dir, "liver")
    rt_structs = resample_structs(rt_structs)
    dice_results = calculate_dice(rt_structs)
    delete_resampled_structs(rt_structs)
    print(dice_results)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
