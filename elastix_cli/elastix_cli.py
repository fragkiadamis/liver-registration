import os
from dotenv import load_dotenv
from utils import validate_paths


def main():
    # Load environmental variables from .env file (create a .env file according to .env.example).
    load_dotenv()

    nifty_dir = os.environ["NIFTY_PATH"]
    elx_cli_output = os.environ["ELX_CLI_OUTPUT_PATH"]
    validate_paths(nifty_dir, elx_cli_output)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
