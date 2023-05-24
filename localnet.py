import os
from time import time
from math import floor

import wandb
from torch.nn import MSELoss
from tqdm import tqdm

from monai.config import print_config
from monai.data import DataLoader, CacheDataset, Dataset
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import Compose, LoadImaged, Resized
from monai.utils import set_determinism
from monai.losses import DiceLoss, MultiScaleLoss
from monai.metrics import DiceMetric

from utils import setup_parser, create_dir, validate_paths
from config.training import *

print_config()

wandb.init(
    project="liver-registration",
    config={
        "architecture": "LocalNet",
        "epochs": 5,
    }
)


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


def forward(batch_data, model, warp_layer):
    fixed_image = batch_data["fixed_image"].to(DEVICE)
    moving_image = batch_data["moving_image"].to(DEVICE)
    moving_label = batch_data["moving_label"].to(DEVICE)

    # predict DDF through LocalNet
    ddf = model(torch.cat((moving_image, fixed_image), dim=1))

    # warp moving image and label with the predicted ddf
    pred_image = warp_layer(moving_image, ddf)
    pred_label = warp_layer(moving_label, ddf)

    return ddf, pred_image, pred_label


def train_model(train_ds, val_ds, train_loader, val_loader):
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

    warp_layer = Warp().to(DEVICE)
    image_loss = MSELoss()
    label_loss = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16])
    optimizer = torch.optim.Adam(localnet.parameters(), lr=INIT_LR)
    train_steps = len(train_ds) // BATCH_SIZE
    val_steps = len(val_ds) // BATCH_SIZE
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    print("[INFO] training the network...")

    metric_values = []
    start_time = time()
    for e in tqdm(range(NUM_EPOCHS)):
        # Set the model to training mode.
        localnet.train()

        total_train_loss, total_val_loss = 0, 0

        # loop over the training set
        for batch_data in train_loader:
            # perform a forward pass and calculate the training loss.
            ddf, pred_image, pred_label = forward(batch_data, localnet, warp_layer)
            pred_label[pred_label > 1] = 1

            fixed_image = batch_data["fixed_image"].to(DEVICE)
            fixed_label = batch_data["fixed_label"].to(DEVICE)
            train_loss = (
                image_loss(pred_image, fixed_image) +
                100 * label_loss(pred_label, fixed_label)
            )

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters.
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # add the loss to the total training loss so far.
            total_train_loss += train_loss.item()

        # switch off autograd
        with torch.no_grad():
            # Set the model to evaluation mode.
            localnet.eval()

            # loop over the validation set
            for val_data in val_loader:
                # make the predictions and calculate the validation loss
                ddf, val_pred_image, val_pred_label = forward(val_data, localnet, warp_layer)
                val_pred_label[val_pred_label > 1] = 1

                val_fixed_image = val_data["fixed_label"].to(DEVICE)
                val_fixed_label = val_data["fixed_label"].to(DEVICE)
                val_loss = (
                    image_loss(val_pred_image, val_fixed_image) +
                    100 * label_loss(val_pred_label, val_fixed_label)
                )
                dice_metric(y_pred=val_pred_label, y=val_fixed_label)

                # add the loss to the total val loss so far.
                total_val_loss += val_loss.item()

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        metric_values.append(metric)

        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_dice": metric})

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Dice: {metric}")

    # display the total time needed to perform the training
    end_time = time()
    print(f"[INFO] total time taken to train the model: {round(((end_time - start_time) / 60) / 60, 2)} hours")
    wandb.finish()


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
    # train_files, val_files = data_dicts[:-idx], data_dicts[-idx:]
    train_files, val_files = data_dicts[:4], data_dicts[4:5]

    set_determinism(seed=0)

    # Cache the transforms of the datasets.
    train_ds = CacheDataset(data=train_files, transform=transforms(), cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_files, transform=transforms(), cache_rate=1.0, num_workers=0)

    # Load the datasets.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False, num_workers=0)

    # Train the LocalNet.
    train_model(train_ds, val_ds, train_loader, val_loader)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
