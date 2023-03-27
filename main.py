import os
from dotenv import load_dotenv
from utils import validate_paths, find_ct_mri_pairs, open_data_frame, calculate_dice, update_dataframe
from preprocessing import extract_from_dicom, resample_mri_2_ct
from elastix_cli import elx_cli_register, apply_transform

# Load environmental variables from .env file (create a .env file according to .env.example).
load_dotenv()


def main():
    # Load environmental variables
    dicom_dir = os.environ["DICOM_PATH"]
    nifty_dir = os.environ["NIFTY_PATH"]
    parameters_path = os.path.abspath(os.environ["ELX_CLI_PARAM_PATH"])
    elx_cli_output = os.path.abspath(os.environ["ELX_CLI_OUTPUT_PATH"])
    # Validate input and output paths
    validate_paths(dicom_dir, nifty_dir)
    validate_paths(nifty_dir, elx_cli_output)

    # Open the dataframe
    df = open_data_frame("output.xlsx")

    # Extract images in nifty format from dicom files, as well as the manual transformations parameters.
    df["Patient"] = extract_from_dicom(dicom_dir, nifty_dir)

    # Calculate initial dice index on the nifty structs.
    # ct_mri_pairs = find_ct_mri_pairs(nifty_dir, studies=["SPECT-CT", "ceMRI"])
    # ct_mri_pairs = resample_mri_2_ct(ct_mri_pairs)
    # initial_dice = calculate_dice(ct_mri_pairs)
    # df = update_dataframe(df, initial_dice, "Initial Dice Index")

    # Register images, apply transformation to mask and recalculate dice index.
    # ct_mri_pairs = find_ct_mri_pairs(nifty_dir, studies=["SPECT-CT", "ceMRI"])
    # ct_mri_pairs = elx_cli_register(ct_mri_pairs, parameters_path, elx_cli_output)
    # new_dice = calculate_dice(ct_mri_pairs)
    # df = update_dataframe(df, new_dice, "Dice Index")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
