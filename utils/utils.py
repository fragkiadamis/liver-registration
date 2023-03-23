import argparse
import sys
import os
from subprocess import run, check_output


# Validate paths, create output paths if they do not exist
def validate_paths(input_dir, output_dir):
    # Validate that arguments are not None.
    if (not input_dir) or (not output_dir):
        sys.exit("Specify input -inp and output -out.")

    # Validate that input directory exists.
    if not os.path.isdir(input_dir):
        sys.exit("This input directory does not exist.")

    # Create output if it does not exist.
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


# Create shell command and execute it.
def execute_shell_cmd(cmd, arguments):
    clitk_command = [os.path.join(os.environ["CLITK_TOOLS_PATH"], cmd)]
    command = clitk_command + arguments
    return check_output(command)
