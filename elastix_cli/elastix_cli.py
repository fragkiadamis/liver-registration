from utils import parse_arguments
from dotenv import load_dotenv
from os import environ as env


def main():
    # Load environmental variables from .env file (create a .env file according to .env.example).
    load_dotenv()

    # Parse CLI arguments and handle inputs and outputs.
    nifty_dir, elx_cli_output = parse_arguments()


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
