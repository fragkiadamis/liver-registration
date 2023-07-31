# Import necessary files and libraries.
import json
import os
from subprocess import run

import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk

from preprocessing import cast_to_type
from utils import calculate_metrics
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


def postprocess_segmentations(input_dir, output_dir, ref_dir):
    results = {}

    folds = os.listdir(input_dir)
    for fold in folds:
        fold_path = os.path.join(input_dir, fold)
        if os.path.isdir(fold_path) and "validation" in fold_path:
            data_path = os.path.join(input_dir, fold, "validation_raw")
            masks = os.listdir(data_path)
            for mask in masks:
                mask_path = os.path.join(data_path, mask)
                cast_to_type([mask_path], "uchar")

                img_in = sitk.ReadImage(mask_path)
                img_npy = sitk.GetArrayFromImage(img_in)
                volume_per_voxel = float(np.prod(img_in.GetSpacing(), dtype=np.float64))

                image, largest_removed, kept_size = remove_redundant_areas(img_npy, [1], volume_per_voxel)

                img_out_itk = sitk.GetImageFromArray(image)
                img_out_itk = copy_geometry(img_out_itk, img_in)

                patient = mask.split(".")[0]
                img_path = os.path.join(output_dir, f"{patient}.nii.gz")
                sitk.WriteImage(img_out_itk, img_path)

                like_img = os.path.join(ref_dir, patient, "spect_ct_volume.nii.gz")
                run(["clitkAffineTransform", "-i", img_path, "-o", img_path, "-l", like_img, "--interp=0"])

                ct_gt_label = os.path.join(ref_dir, patient, "spect_ct_liver.nii.gz")
                results.update({
                    patient: {
                        "liver": {
                            "U-NET3D": calculate_metrics(ct_gt_label, img_path)
                        }
                    }
                })
                print(patient, results[patient])

        with open(f"{output_dir}/../unet3d_evaluation.json", "w") as fp:
            json.dump(results, fp)


# Resample images to 512x512x448 after deep learning registration.
def postprocess_registrations(input_dir, output_dir, ref_dir):
    localnet_items = os.listdir(input_dir)
    output_dir = create_dir(input_dir, output_dir)

    for item in localnet_items:
        item_path = os.path.join(input_dir, item)

        if os.path.isdir(item_path) and "fold" in item_path:
            for patient in os.listdir(item_path):
                if patient == "JaneDoe_ANON12304":
                    continue

                print(f"-Patient: {patient}")
                patient_dir = os.path.join(item_path, patient)
                patient_output_dir = create_dir(output_dir, patient)

                mri_volume_pred = os.path.join(patient_dir, "mri_volume_pred.nii.gz")
                mri_liver_pred = os.path.join(patient_dir, "mri_liver_pred.nii.gz")
                mri_tumor_pred = os.path.join(patient_dir, "mri_tumor_pred.nii.gz")
                mri_tumor_bbox_pred = os.path.join(patient_dir, "mri_tumor_bbox_pred.nii.gz")
                mri_ddf_pred = os.path.join(patient_dir, "mri_ddf_pred.nii.gz")

                mri_volume_out = os.path.join(patient_output_dir, "mri_volume_pred.nii.gz")
                mri_liver_out = os.path.join(patient_output_dir, "mri_liver_pred.nii.gz")
                mri_tumor_out = os.path.join(patient_output_dir, "mri_tumor_pred.nii.gz")
                mri_tumor_bbox_out = os.path.join(patient_output_dir, "mri_tumor_bbox_pred.nii.gz")
                mri_ddf_out = os.path.join(patient_output_dir, "mri_ddf_pred.nii.gz")

                ct_volume = os.path.join(ref_dir, patient, "spect_ct_volume.nii.gz")
                ct_liver = os.path.join(ref_dir, patient, "spect_ct_liver.nii.gz")
                ct_tumor = os.path.join(ref_dir, patient, "spect_ct_tumor.nii.gz")
                ct_tumor_bbox = os.path.join(ref_dir, patient, "spect_ct_tumor_bbox.nii.gz")

                print("\t-Resampling...")
                arg_list = [
                    "clitkAffineTransform", "-i", mri_volume_pred, "-o", mri_volume_out, "--interp=1", "-l", ct_volume
                ]
                run(arg_list)
                arg_list = [
                    "clitkAffineTransform", "-i", mri_liver_pred, "-o", mri_liver_out, "--interp=0", "-l", ct_volume
                ]
                run(arg_list)
                arg_list = [
                    "clitkAffineTransform", "-i", mri_tumor_pred, "-o", mri_tumor_out, "--interp=0", "-l", ct_volume
                ]
                run(arg_list)
                arg_list = [
                    "clitkAffineTransform", "-i", mri_tumor_bbox_pred, "-o", mri_tumor_bbox_out, "--interp=0", "-l", ct_volume
                ]
                run(arg_list)
                arg_list = [
                    "clitkAffineTransform", "-i", mri_ddf_pred, "-o", mri_ddf_out, "--interp=1", "-l", ct_volume
                ]
                run(arg_list)

                print("\t-Calculating results...")
                results = {
                    "liver": calculate_metrics(ct_liver, mri_liver_out),
                    "tumor": calculate_metrics(ct_tumor, mri_tumor_out),
                    "tumor_bbox": calculate_metrics(ct_tumor_bbox, mri_tumor_bbox_out)
                }

                with open(f"{patient_output_dir}/localnet_evaluation.json", "w") as fp:
                    json.dump(results, fp)


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser("config/postprocessing_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = create_dir(dir_name, args.o)
    ref_dir = os.path.join(dir_name, args.ref)
    postprocessing_type = args.t

    if postprocessing_type == "seg":
        postprocess_segmentations(input_dir, output_dir, ref_dir)
    elif postprocessing_type == "reg":
        postprocess_registrations(input_dir, output_dir, ref_dir)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
