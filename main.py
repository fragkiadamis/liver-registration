from dotenv import load_dotenv
from os import environ as env
from utils import validate_paths
from dcm_utils import extract_from_dicom

# Load environmental variables from .env file (create a .env file according to .env.example).
load_dotenv()


def main():
    dicom_dir = env["DICOM_PATH"]
    nifty_dir = env["NIFTY_PATH"]
    elx_cli_output = env["ELX_CLI_OUTPUT_PATH"]

    validate_paths(dicom_dir, nifty_dir)
    validate_paths(nifty_dir, elx_cli_output)

    extract_from_dicom(dicom_dir, nifty_dir)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
