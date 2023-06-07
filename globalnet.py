import os
import random
from time import time
from math import floor

import torch
import wandb
from tqdm import tqdm
import numpy as np

from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.networks.blocks import Warp
from monai.networks.nets import GlobalNet
from monai.transforms import Compose, LoadImaged, Resized
from monai.losses import DiceLoss, MultiScaleLoss
from monai.metrics import DiceMetric

from utils import setup_parser, create_dir, validate_paths
from config.training import ARCHITECTURE, NUM_EPOCHS, INIT_LR, BATCH_SIZE, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, \
    INPUT_IMAGE_DEPTH, DEVICE, PIN_MEMORY, TRAIN_SPLIT, WD

print_config()

wandb.init(
    project="liver-registration",
    name="globalnet",
    config={
        "architecture": ARCHITECTURE,
        "epochs": NUM_EPOCHS,
        "learning_rate": INIT_LR,
        "batch_size": BATCH_SIZE
    }
)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Preprocessing transforms.
def transforms():
    return Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("trilinear", "trilinear", "nearest", "nearest"),
                align_corners=(True, True, None, None),
                spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_DEPTH),
            ),
        ]
    )


# Forward the input to the model to get the prediction.
def forward(sample, model, warp_layer):
    x_fixed = sample["fixed_image"].to(DEVICE)
    x_moving = sample["moving_image"].to(DEVICE)
    y_moving = sample["moving_label"].to(DEVICE)

    ddf = model(torch.cat((x_fixed, x_moving), dim=1))
    y_pred = warp_layer(y_moving, ddf)

    return ddf, y_pred


# Train the model.
def train(model, train_loader, criterion, optimizer, warp_layer):
    model.train()
    total_train_loss = 0

    # loop over the training set.
    for batch_data in train_loader:
        ddf, y_pred = forward(batch_data, model, warp_layer)
        fixed_label = batch_data["fixed_label"].to(DEVICE)
        y_pred[y_pred > 1] = 1

        optimizer.zero_grad()
        train_loss = criterion(y_pred, fixed_label)
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()

    return total_train_loss


# Validate the model.
def validate(model, val_loader, criterion, warp_layer, dice_metric):
    model.eval()
    total_val_loss = 0

    # Loop over the validation set.
    with torch.no_grad():
        for val_data in val_loader:
            ddf, y_pred = forward(val_data, model, warp_layer)
            fixed_label = val_data["fixed_label"].to(DEVICE)
            y_pred[y_pred > 1] = 1

            val_loss = criterion(y_pred, fixed_label)
            total_val_loss += val_loss.item()

            dice = dice_metric(y_pred=y_pred, y=fixed_label)
            # print(f"Dice: {dice[0][0]}")

        dice_avg = dice_metric.aggregate().item()
        dice_metric.reset()

    return total_val_loss, dice_avg


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/dl_registration_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    create_dir(dir_name, output_dir)

    model_path = os.path.join(output_dir, args.n)

    # Load dataset file paths.
    data = [
        {
            "fixed_image": os.path.join(input_dir, f"images/{patient}_fixed.nii.gz"),
            "moving_image": os.path.join(input_dir, f"images/{patient}_moving.nii.gz"),
            "fixed_label": os.path.join(input_dir, f"labels/{patient}_fixed.nii.gz"),
            "moving_label": os.path.join(input_dir, f"labels/{patient}_moving.nii.gz"),
        }
        for patient in os.listdir("data/mri_spect_nii_iso")
    ]

    # Split training and validation sets.
    train_size = floor(len(data) * TRAIN_SPLIT)
    train_files, val_files = data[:train_size], data[train_size:]

    # Cache the transforms of the datasets.
    train_ds = CacheDataset(data=train_files, transform=transforms(), cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_files, transform=transforms(), cache_rate=1.0, num_workers=0)

    # Load the datasets.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False, num_workers=0)

    # Create LocalNet, losses and optimizer.
    globalnet = GlobalNet(
        depth=4,
        image_size=[INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_DEPTH],
        spatial_dims=3,
        in_channels=2,
        num_channel_initial=16,
        out_kernel_initializer="zeros",
    ).to(DEVICE)

    train_steps = len(train_ds) // BATCH_SIZE
    val_steps = len(val_ds) // BATCH_SIZE

    warp_layer = Warp(mode="bilinear", padding_mode="border").to(DEVICE)
    criterion = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16, 32])
    optimizer = torch.optim.Adam(globalnet.parameters(), lr=INIT_LR, weight_decay=WD)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    dice_values = []
    best_dice = 0

    start_time = time()
    for e in tqdm(range(NUM_EPOCHS)):
        train_loss = train(globalnet, train_loader, criterion, optimizer, warp_layer)
        val_loss, val_dice_avg = validate(globalnet, val_loader, criterion, warp_layer, dice_metric)

        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Dice Index: {val_dice_avg}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "dice": val_dice_avg}, step=e+1)

        dice_values.append(val_dice_avg)
        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            torch.save(globalnet.state_dict(), f"{model_path}_best.pth")
            print(f"[INFO] saved model in epoch {e + 1} as new best model.")

        wandb.log({"best_dice": best_dice}, step=e+1)

    end_time = time()
    print(f"[INFO] total time taken to train the model: {round(((end_time - start_time) / 60) / 60, 2)} hours")
    wandb.finish()


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
