import os
import random
from time import time
from math import floor

import torch
import wandb
from tqdm import tqdm
import numpy as np
import nibabel as nib

from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet, GlobalNet
from monai.transforms import Compose, LoadImaged, Resized
from monai.losses import BendingEnergyLoss, DiceLoss, MultiScaleLoss
from monai.metrics import DiceMetric

from utils import setup_parser, create_dir, validate_paths
from config.training import ARCHITECTURE, NUM_EPOCHS, INIT_LR, BATCH_SIZE, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, \
    INPUT_IMAGE_DEPTH, DEVICE, PIN_MEMORY, TRAIN_SPLIT

print_config()


# Preprocessing transforms.
def transforms():
    return Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
            # Resized(
            #     keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
            #     mode=("trilinear", "trilinear", "nearest", "nearest"),
            #     align_corners=(True, True, None, None),
            #     spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_DEPTH),
            # ),
        ]
    )


# Forward the input to the model to get the prediction.
def forward(sample, model, warp_layer, warp_layer_nn, inference=False):
    x_fixed = sample["fixed_image"].to(DEVICE)
    x_moving = sample["moving_image"].to(DEVICE)
    y_moving = sample["moving_label"].to(DEVICE)

    ddf = model(torch.cat((x_fixed, x_moving), dim=1))
    y_pred = warp_layer_nn(y_moving, ddf)

    if inference:
        x_pred = warp_layer(x_moving, ddf)
        return ddf, x_pred, y_pred

    return ddf, y_pred


def initial_dice(test_loader, dice_metric):
    for test_data in test_loader:
        fixed_label = test_data["fixed_label"].to(DEVICE)
        moving_label = test_data["moving_label"].to(DEVICE)

        dice = dice_metric(y_pred=moving_label, y=fixed_label)
        print(dice[0][0])

    dice_avg = dice_metric.aggregate().item()
    dice_metric.reset()

    return dice_avg


# Inference the model on the testing data.
def inference_model(model, test_loader, warp_layer, warp_layer_nn, dice_metric):
    model.eval()

    # Loop over the validation set.
    with torch.no_grad():
        predictions = []
        for test_data in test_loader:
            ddf, x_pred, y_pred = forward(test_data, model, warp_layer, warp_layer_nn, inference=True)
            fixed_label = test_data["fixed_label"].to(DEVICE)

            dice = dice_metric(y_pred=y_pred, y=fixed_label)
            print(dice[0][0])

            predictions.append({
                test_data["patient"][0]: {
                    "image_tensor": x_pred,
                    "label_tensor": y_pred
                }
            })

        dice_avg = dice_metric.aggregate().item()
        dice_metric.reset()

    return dice_avg, predictions


def save_predictions(predictions, output):
    for prediction in predictions:
        patient = list(prediction.keys())[0]
        image = prediction[patient]["image_tensor"].cpu().numpy()[0, 0]
        image = np.flip(image, axis=(0, 1))
        image = nib.Nifti1Image(image, affine=np.eye(4))

        label = prediction[patient]["label_tensor"].cpu().numpy()[0, 0]
        label = np.flip(label, axis=(0, 1))
        label = nib.Nifti1Image(label, affine=np.eye(4))

        nib.save(image, f"{output}/{patient}_volume_pred.nii.gz")
        nib.save(label, f"{output}/{patient}_liver_pred.nii.gz")


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/inference_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)
    model_path = os.path.join(dir_name, args.m)

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    create_dir(dir_name, output_dir)

    data = [
        {
            "fixed_image": os.path.join(input_dir, f"images/{patient}_fixed.nii.gz"),
            "moving_image": os.path.join(input_dir, f"images/{patient}_moving.nii.gz"),
            "fixed_label": os.path.join(input_dir, f"labels/{patient}_fixed.nii.gz"),
            "moving_label": os.path.join(input_dir, f"labels/{patient}_moving.nii.gz"),
            "patient": patient
        }
        for patient in os.listdir("data/mri_spect_nii_iso")
    ]

    train_size = floor(len(data) * TRAIN_SPLIT)
    test_files = data[train_size:]
    test_ds = CacheDataset(data=test_files, transform=transforms(), cache_rate=1.0, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False, num_workers=0)

    model = GlobalNet(
        depth=5,
        image_size=[INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_DEPTH],
        spatial_dims=3,
        in_channels=2,
        num_channel_initial=32,
        out_activation=None,
        out_kernel_initializer="zeros",
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    warp_layer = Warp().to(DEVICE)
    warp_layer_nn = Warp(mode="nearest").to(DEVICE)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    init_dice = initial_dice(test_loader, dice_metric)
    print(f"Average Initial Dice: {init_dice}")
    avg_dice, predictions = inference_model(model, test_loader, warp_layer, warp_layer_nn, dice_metric)
    print(f"Average Final Dice: {avg_dice}")

    save_predictions(predictions, output_dir)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
