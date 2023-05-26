import os
from time import time
from math import floor

import wandb
from tqdm import tqdm

from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import Compose, LoadImaged, Resized
from monai.losses import BendingEnergyLoss, DiceLoss, MultiScaleLoss
from monai.metrics import DiceMetric

from utils import setup_parser, create_dir, validate_paths
from config.training import *

print_config()

wandb.init(
    project="liver-registration",
    config={
        "architecture": "LocalNet",
        "epochs": NUM_EPOCHS,
    }
)


# Compare the parameters in two models to see if they are the same or not.
def are_equal(model_1, model_2):
    parameters_equal = []
    for (m1p_name, m1_param), (m2p_name, m2_param) in zip(model_1.items(), model_2.items()):
        equal = torch.equal(m1_param, m2_param)
        parameters_equal.append(equal)

    # Check if all parameters are equal
    all_parameters_equal = all(parameters_equal)

    return True if all_parameters_equal else False


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


def forward(sample, model, warp_layer):
    x_fixed = sample["fixed_image"].to(DEVICE)
    x_moving = sample["moving_image"].to(DEVICE)
    y_moving = sample["moving_label"].to(DEVICE)

    ddf = model(torch.cat((x_fixed, x_moving), dim=1))
    y_pred = warp_layer(y_moving, ddf)

    return ddf, y_pred


def train(model, train_loader, criterion, optimizer, warp_layer, regularization):
    model.train()
    total_train_loss = 0

    # loop over the training set.
    for batch_data in train_loader:
        ddf, y_pred = forward(batch_data, model, warp_layer)
        fixed_label = batch_data["fixed_label"].to(DEVICE)
        y_pred[y_pred > 1] = 1

        train_loss = criterion(y_pred, fixed_label) + 0.5 * regularization(ddf)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()

    return total_train_loss


def validate(model, val_loader, criterion, warp_layer, regularization, dice_metric):
    model.eval()
    total_val_loss = 0

    # Loop over the validation set.
    with torch.no_grad():
        for val_sample in val_loader:
            ddf, y_pred = forward(val_sample, model, warp_layer)
            fixed_label = val_sample["fixed_label"].to(DEVICE)
            y_pred[y_pred > 1] = 1

            val_loss = criterion(y_pred, fixed_label) + 0.5 * regularization(ddf)
            total_val_loss += val_loss.item()
            dice_metric(y_pred=y_pred, y=fixed_label)

        dice_idx = dice_metric.aggregate().item()
        dice_metric.reset()

    return total_val_loss, dice_idx


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/dl_registration_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    create_dir(dir_name, output_dir)

    # Load dataset file paths.
    data_dicts = [
        {
            "fixed_image": os.path.join(input_dir, f"images/{patient}_fixed.nii.gz"),
            "moving_image": os.path.join(input_dir, f"images/{patient}_moving.nii.gz"),
            "fixed_label": os.path.join(input_dir, f"labels/{patient}_fixed.nii.gz"),
            "moving_label": os.path.join(input_dir, f"labels/{patient}_moving.nii.gz"),
        }
        for patient in os.listdir("data/mri_spect_nii_iso")
    ]

    # Split training and validation sets.
    idx = floor(len(data_dicts) * VAL_SPLIT)
    train_files, val_files = data_dicts[:-idx], data_dicts[-idx:]
    # train_files, val_files = data_dicts[:5], data_dicts[5:6]

    # Cache the transforms of the datasets.
    train_ds = CacheDataset(data=train_files, transform=transforms(), cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_files, transform=transforms(), cache_rate=1.0, num_workers=0)

    # Load the datasets.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False, num_workers=0)

    # Create LocalNet, losses and optimizer.
    localnet = LocalNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        num_channel_initial=32,
        extract_levels=(3,),
        out_activation=None,
        out_kernel_initializer="zeros",
    ).to(DEVICE)

    train_steps = len(train_ds) // BATCH_SIZE
    val_steps = len(val_ds) // BATCH_SIZE

    warp_layer = Warp().to(DEVICE)
    criterion = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16, 32])
    regularization = BendingEnergyLoss()
    optimizer = torch.optim.Adam(localnet.parameters(), lr=INIT_LR)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    start_time = time()
    for e in tqdm(range(NUM_EPOCHS)):
        train_loss = train(localnet, train_loader, criterion, optimizer, warp_layer, regularization)
        val_loss, dice_idx = validate(localnet, val_loader, criterion, warp_layer, regularization, dice_metric)

        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Dice Index: {dice_idx}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "dice_index": dice_idx})

    end_time = time()
    print(f"[INFO] total time taken to train the model: {round(((end_time - start_time) / 60) / 60, 2)} hours")
    wandb.finish()


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
