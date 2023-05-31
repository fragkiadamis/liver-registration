import os
from math import floor
from time import time
from matplotlib import pyplot as plt

import torch
import wandb
from tqdm import tqdm

from monai.config import print_config
from monai.apps import download_url
from monai.data import CacheDataset, DataLoader
from monai.losses import MultiScaleLoss, BendingEnergyLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import Compose, LoadImaged, Resized

from config.training import DEVICE, BATCH_SIZE, PIN_MEMORY, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_DEPTH, \
    VAL_SPLIT, INIT_LR, NUM_EPOCHS, ARCHITECTURE, TRAIN_SPLIT
from utils import setup_parser, validate_paths, create_dir

print_config()

wandb.init(
    project="liver-registration",
    tags=["registration"],
    config={
        "architecture": ARCHITECTURE,
        "epochs": NUM_EPOCHS,
        "learning_rate": INIT_LR,
        "batch_size": BATCH_SIZE
    }
)


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
def forward(sample, model, warp_layer, inference=False):
    x_fixed = sample["fixed_image"].to(DEVICE)
    x_moving = sample["moving_image"].to(DEVICE)
    y_moving = sample["moving_label"].to(DEVICE)

    ddf = model(torch.cat((x_fixed, x_moving), dim=1))
    y_pred = warp_layer(y_moving, ddf)

    if inference:
        x_pred = warp_layer(x_moving, ddf)
        return ddf, x_pred, y_pred

    return ddf, y_pred


# Train the model.
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


# Validate the model.
def validate(model, val_loader, criterion, warp_layer, regularization, dice_metric):
    model.eval()
    total_val_loss = 0

    # Loop over the validation set.
    with torch.no_grad():
        for val_data in val_loader:
            ddf, y_pred = forward(val_data, model, warp_layer)
            fixed_label = val_data["fixed_label"].to(DEVICE)

            val_loss = criterion(y_pred, fixed_label) + 0.5 * regularization(ddf)
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

    model_path = os.path.join(output_dir, f"{args.n}.pth")

    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/pair_lung_ct.pth"
    download_url(resource, model_path)

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
    val_size = train_size + floor(len(data) * VAL_SPLIT)
    train_files, val_files = data[:train_size], data[train_size:]
    # train_files, val_files, test_files = data[:1], data[1:2], data[2:3]

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

    localnet.load_state_dict(torch.load(model_path))

    train_steps = len(train_ds) // BATCH_SIZE
    val_steps = len(val_ds) // BATCH_SIZE

    warp_layer = Warp().to(DEVICE)
    warp_layer_nn = Warp("nearest").to(DEVICE)
    criterion = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16, 32])
    regularization = BendingEnergyLoss()
    optimizer = torch.optim.Adam(localnet.parameters(), lr=INIT_LR)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    dice_values = []
    best_dice = 0

    start_time = time()
    for e in tqdm(range(NUM_EPOCHS)):
        train_loss = train(localnet, train_loader, criterion, optimizer, warp_layer, regularization)
        val_loss, val_dice_avg = validate(localnet, val_loader, criterion, warp_layer, regularization, dice_metric)

        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Dice Index: {val_dice_avg}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "dice_index": val_dice_avg})

        dice_values.append(val_dice_avg)
        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            torch.save(localnet.state_dict(), f"{model_path}_best.pth")
            print(f"[INFO] saved model in epoch {e + 1} as new best model.")

    end_time = time()
    print(f"[INFO] total time taken to train the model: {round(((end_time - start_time) / 60) / 60, 2)} hours")
    wandb.finish()

    # localnet.eval()
    # with torch.no_grad():
    #     for i, val_data in enumerate(val_loader):
    #         val_ddf, val_pred_image, val_pred_label = forward(val_data, localnet, warp_layer_nn, inference=True)
    #         val_fixed_label = val_data["fixed_label"].to(DEVICE)
    #
    #         dice = dice_metric(y_pred=val_pred_label, y=val_fixed_label)
    #         print(f"Dice: {dice[0][0]}")
    #
    #         val_pred_image = val_pred_image.cpu().numpy()[0, 0].transpose((1, 0, 2))
    #         val_pred_label = val_pred_label.cpu().numpy()[0, 0].transpose((1, 0, 2))
    #         val_moving_image = val_data["moving_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
    #         val_moving_label = val_data["moving_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))
    #         val_fixed_image = val_data["fixed_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
    #         val_fixed_label = val_fixed_label.cpu().numpy()[0, 0].transpose((1, 0, 2))
    #
    #         # for depth in range(10):
    #         #     depth = depth * 10
    #         #     plt.figure("check", (18, 6))
    #         #     plt.subplot(1, 6, 1)
    #         #     plt.title(f"moving_image {i} d={depth}")
    #         #     plt.imshow(val_moving_image[:, :, depth], cmap="gray")
    #         #     plt.subplot(1, 6, 2)
    #         #     plt.title(f"moving_label {i} d={depth}")
    #         #     plt.imshow(val_moving_label[:, :, depth])
    #         #     plt.subplot(1, 6, 3)
    #         #     plt.title(f"fixed_image {i} d={depth}")
    #         #     plt.imshow(val_fixed_image[:, :, depth], cmap="gray")
    #         #     plt.subplot(1, 6, 4)
    #         #     plt.title(f"fixed_label {i} d={depth}")
    #         #     plt.imshow(val_fixed_label[:, :, depth])
    #         #     plt.subplot(1, 6, 5)
    #         #     plt.title(f"pred_image {i} d={depth}")
    #         #     plt.imshow(val_pred_image[:, :, depth], cmap="gray")
    #         #     plt.subplot(1, 6, 6)
    #         #     plt.title(f"pred_label {i} d={depth}")
    #         #     plt.imshow(val_pred_label[:, :, depth])
    #         #     plt.show()
    #
    #     dice_avg = dice_metric.aggregate().item()
    #     dice_metric.reset()
    #
    #     print(dice_avg)


if __name__ == "__main__":
    main()
