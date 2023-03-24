import os
from dotenv import load_dotenv
from utils import execute_shell_cmd, find_ct_mri_pair


# Calculate the dice index between 2 RT structures
def calculate_dice(rt_structs):
    dice_results = {}

    for patient in rt_structs:
        ct_struct = rt_structs[patient][0]
        mri_struct = rt_structs[patient][1]
        dice_index = execute_shell_cmd("clitkDice", ["-i", ct_struct, "-j", mri_struct])
        dice_index = float(dice_index.decode("utf-8"))

        print(f"{patient} Dice Index: {dice_index}")
        dice_results[f"{patient}"] = dice_index

    return dice_results


def main():
    # Load environmental variables from .env file (create a .env file according to .env.example).
    load_dotenv()

    # Parse CLI arguments and handle inputs and outputs.
    nifty_dir = os.environ["NIFTY_PATH"]

    # Calculate initial overlap between.
    rt_structs = find_ct_mri_pair(nifty_dir, studies=["SPECT-CT", "ceMRI"], filename="rtstruct_liver_resampled_to_ct")
    dice_results = calculate_dice(rt_structs)
    print(dice_results)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
