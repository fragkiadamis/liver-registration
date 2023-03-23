import os
from dotenv import load_dotenv
from utils import validate_paths
from dcm_utils import extract_from_dicom
from reg_evaluation import find_respective_structs, resample_structs, calculate_dice, delete_resampled_structs

# Load environmental variables from .env file (create a .env file according to .env.example).
load_dotenv()


def main():
    dicom_dir = os.environ["DICOM_PATH"]
    nifty_dir = os.environ["NIFTY_PATH"]
    elx_cli_output = os.environ["ELX_CLI_OUTPUT_PATH"]

    validate_paths(dicom_dir, nifty_dir)
    validate_paths(nifty_dir, elx_cli_output)

    # Extract images in nifty format from dicom files, as well as the manual transformations parameters
    # extract_from_dicom(dicom_dir, nifty_dir)

    # Calculate initial dice index on the nifty structs.
    rt_structs = find_respective_structs(nifty_dir, "liver")
    rt_structs = resample_structs(rt_structs)
    initial_dice_indices = calculate_dice(rt_structs)
    delete_resampled_structs(rt_structs)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
