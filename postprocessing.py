# Import necessary files and libraries.
import os
from subprocess import run

import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk

from preprocessing import cast_to_type
from utils import setup_parser, create_dir


def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


def remove_redundant_areas(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                           minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param volume_per_voxel:
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size


def resample_mask(mask, like):
    pass


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("config/preprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = create_dir(dir_name, args.o)

    masks = os.listdir(input_dir)
    for idx, mask in enumerate(masks):
        print(f"Processing {mask}")
        mask_path = os.path.join(input_dir, mask)
        cast_to_type([mask_path], "uchar")

        img_in = sitk.ReadImage(mask_path)
        img_npy = sitk.GetArrayFromImage(img_in)
        volume_per_voxel = float(np.prod(img_in.GetSpacing(), dtype=np.float64))

        image, largest_removed, kept_size = remove_redundant_areas(img_npy, [1], volume_per_voxel)

        img_out_itk = sitk.GetImageFromArray(image)
        img_out_itk = copy_geometry(img_out_itk, img_in)

        patient = mask.split(".")[0]
        patient_output = os.path.join(output_dir, patient)

        img_path = os.path.join(patient_output, "ct_auto_generated_liver.nii.gz")
        sitk.WriteImage(img_out_itk, img_path)

        like_img = os.path.join(patient_output, "ct_volume.nii.gz")
        run(["clitkAffineTransform", "-i", img_path, "-o", img_path, "-l", like_img, "--interp=0"])


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
