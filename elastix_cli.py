import os
from dcm import extract_from_dicom


def main():
    # Set input and output directories.
    dicom_dir, nifty_dir = "dicom", "nifty"

    # If nifty output directory does not exist, create it
    if not os.path.isdir(nifty_dir):
        os.mkdir(nifty_dir)
        # Convert dicom data to nifty and extract the manual transformations.
        extract_from_dicom(dicom_dir, nifty_dir)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
