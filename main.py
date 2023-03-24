import os
from dotenv import load_dotenv
from utils import validate_paths, find_ct_mri_pair
from preprocessing import extract_from_dicom, resample_mri_2_ct
from reg_evaluation import calculate_dice
import pandas as pd

# Load environmental variables from .env file (create a .env file according to .env.example).
load_dotenv()


def main():
    dicom_dir = os.environ["DICOM_PATH"]
    nifty_dir = os.environ["NIFTY_PATH"]
    elx_cli_output = os.environ["ELX_CLI_OUTPUT_PATH"]

    validate_paths(dicom_dir, nifty_dir)
    validate_paths(nifty_dir, elx_cli_output)

    df = pd.DataFrame()
    # Open the file with all the data (if it exists).
    if "output.xlsx" in os.listdir("./"):
        df = pd.read_excel("output.xlsx", index_col=0)

    # Extract images in nifty format from dicom files, as well as the manual transformations parameters.
    # df["Patient"] = extract_from_dicom(dicom_dir, nifty_dir)

    # Calculate initial dice index on the nifty structs.
    rt_structs = find_ct_mri_pair(nifty_dir, studies=["SPECT-CT", "ceMRI"], filename="rtstruct_liver")
    rt_structs = resample_mri_2_ct(rt_structs)
    initial_dice = calculate_dice(rt_structs)

    # Assign corresponding indices to the dataframe
    for patient in initial_dice:
        df.loc[df["Patient"] == patient, "Initial Liver Overlap (Dice Index)"] = initial_dice[patient]

    # Save the data
    df.to_excel("output.xlsx")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
