# Import necessary files and libraries.
import os
from preprocessing import cast_to_type
from utils import setup_parser


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("parser/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)

    masks = os.listdir(input_dir)
    for idx, mask in enumerate(masks):
        masks[idx] = os.path.join(input_dir, mask)

    cast_to_type(masks, "uchar")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
