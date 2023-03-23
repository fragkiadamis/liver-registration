import os
from dcm import extract_from_dicom
from dotenv import load_dotenv

# Load environmental variables from .env file (create a .env file according to .env.example)
load_dotenv()


def main():
    dicom_dir = os.environ.get("DICOM_PATH")
    nifty_dir = os.environ.get("NIFTY_PATH")
    elx_cli_output = os.environ.get("ELX_CLI_OUTPUT_PATH")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
