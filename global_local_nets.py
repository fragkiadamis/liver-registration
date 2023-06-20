import os
import random
from time import time
from math import floor

import torch
from torch.optim import Adam

import wandb
from tqdm import tqdm
import numpy as np
import nibabel as nib

from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.networks.blocks import Warp
from monai.networks.nets import GlobalNet, LocalNet
from monai.transforms import Compose, LoadImaged, Resized
from monai.losses import DiceLoss, MultiScaleLoss, BendingEnergyLoss
from monai.metrics import DiceMetric

from utils import setup_parser, create_dir, validate_paths
from config.training import ARCHITECTURE, NUM_EPOCHS, INIT_LR, BATCH_SIZE, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, \
    INPUT_IMAGE_DEPTH, DEVICE, PIN_MEMORY, TRAIN_SPLIT, WD

print_config()

wandb.init(
    project="liver-registration",
    name=ARCHITECTURE,
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


# Print the time logs.
def print_time_logs(timer):
    print()
    for key, value in timer.items():
        print(f"[INFO] {key} Time: {round(value / 60, 2)} minutes.")
    print()


# Extract the images from the torches and save them.
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


# Calculate the initial dice average on the data.
def initial_dice(test_loader, dice_metric):
    for test_data in test_loader:
        fixed_label = test_data["fixed_label"].to(DEVICE)
        moving_label = test_data["moving_label"].to(DEVICE)

        dice = dice_metric(y_pred=moving_label, y=fixed_label)
        print(dice[0][0])

    dice_avg = dice_metric.aggregate().item()
    dice_metric.reset()

    return dice_avg


# Initialize the model that according to the argument.
def initialize_model(model_name):
    if model_name == "globalnet":
        return GlobalNet(
            depth=5,
            image_size=[INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_DEPTH],
            spatial_dims=3,
            in_channels=2,
            num_channel_initial=32,
            out_activation=None,
            out_kernel_initializer="zeros",
        ).to(DEVICE)

    elif model_name == "localnet":
        return LocalNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=32,
            extract_levels=(3,),
            out_activation=None,
            out_kernel_initializer="zeros",
        ).to(DEVICE)


# Preprocessing and data augmentation.
def transforms():
    return Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
        ]
    )


# Forward the input to the model to get the prediction.
def forward(sample, model, warp_layer, inference=False):
    x_fixed = sample["fixed_image"].to(DEVICE)
    x_moving = sample["moving_image"].to(DEVICE)
    y_moving = sample["moving_label"].to(DEVICE)

    # Predict DDF from fixed and moving images
    ddf = model(torch.cat((x_fixed, x_moving), dim=1))

    # For inference, Warp labels using nearest neighbor.
    if inference:
        y_pred = warp_layer["binary"](y_moving, ddf)
        x_pred = warp_layer["linear"](x_moving, ddf)
        return ddf, x_pred, y_pred

    # Interpolate using linear interpolation in order for the network to perform backpropagation.
    y_pred = warp_layer(y_moving, ddf)

    return ddf, y_pred


# Train the model.
def train(model, train_loader, criterion, regularization, optimizer, warp_layer):
    model.train()
    total_train_loss = 0

    timer = {"Training": 0, "Loading": 0, "Forward": 0, "Loss": 0, "Backpropagation": 0}
    start_time = time()

    # loop over the training set.
    for batch_data in train_loader:
        load_time = time()

        # Predict DDF and moving label.
        ddf, y_pred = forward(batch_data, model, warp_layer["linear"])
        fwd_calc = time()

        fixed_label = batch_data["fixed_label"].to(DEVICE)
        y_pred[y_pred > 1] = 1

        optimizer.zero_grad()

        # Calculate loss, apply regularization and perform backpropagation.
        train_loss = criterion(y_pred, fixed_label)
        loss_calc = time()

        if regularization:
            train_loss += 0.5 * regularization(ddf)

        back_calc = time()
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()

        timer["Training"] += time() - start_time
        timer["Loading"] += load_time - start_time
        timer["Forward"] += fwd_calc - load_time
        timer["Loss"] += loss_calc - fwd_calc
        timer["Backpropagation"] += back_calc - loss_calc

        start_time = time()

    print_time_logs(timer)

    return total_train_loss


# Validate the model.
def validate(model, val_loader, criterion, regularization, warp_layer, dice_metric):
    model.eval()
    total_val_loss = 0

    timer = {"Validation": 0, "Loading": 0, "Forward": 0, "Loss": 0, "Dice": 0}
    start_time = time()

    # Loop over the validation set.
    with torch.no_grad():
        for val_data in val_loader:
            load_time = time()

            # Predict DDF and moving label.
            ddf, y_pred = forward(val_data, model, warp_layer["binary"])
            fwd_calc = time()

            fixed_label = val_data["fixed_label"].to(DEVICE)
            y_pred[y_pred > 1] = 1

            # Calculate loss, apply regularization it exists.
            val_loss = criterion(y_pred, fixed_label)
            loss_calc = time()

            if regularization:
                val_loss += 0.5 * regularization(ddf)
            total_val_loss += val_loss.item()

            # Calculate dice.
            dice_metric(y_pred=y_pred, y=fixed_label)
            dice_calc = time()

            timer["Validation"] += time() - start_time
            timer["Loading"] += load_time - start_time
            timer["Forward"] += fwd_calc - load_time
            timer["Loss"] += loss_calc - fwd_calc
            timer["Dice"] += dice_calc - loss_calc

            start_time = time()

        # Aggregate dice results.
        dice_avg = dice_metric.aggregate().item()
        dice_metric.reset()

    print_time_logs(timer)

    return total_val_loss, dice_avg


# Inference the model.
def inference_model(model, test_loader, warp_layer, dice_metric):
    model.eval()

    # Loop over the validation set.
    with torch.no_grad():
        predictions = []
        for test_data in test_loader:
            # Predict moving image and moving label.
            ddf, x_pred, y_pred = forward(test_data, model, warp_layer, inference=True)
            fixed_label = test_data["fixed_label"].to(DEVICE)

            # Calculate dice on the predicted label.
            dice = dice_metric(y_pred=y_pred, y=fixed_label)
            print(dice[0][0])

            # Append the predictions for the respective patient.
            predictions.append({
                test_data["patient"][0]: {
                    "image_tensor": x_pred,
                    "label_tensor": y_pred
                }
            })

        # Aggregate dice results.
        dice_avg = dice_metric.aggregate().item()
        dice_metric.reset()

    return dice_avg, predictions


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/dl_registration_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)
    model_name = args.n

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    create_dir(dir_name, output_dir)

    # Initialize the model.
    model_path = os.path.join(output_dir, args.n)

    # Load dataset file paths.
    data = [
        {
            "fixed_image": os.path.join(input_dir, patient, "ct_volume.nii.gz"),
            "moving_image": os.path.join(input_dir, patient, "mri_volume.nii.gz"),
            "fixed_label": os.path.join(input_dir, patient, "ct_liver.nii.gz"),
            "moving_label": os.path.join(input_dir, patient, "mri_liver.nii.gz"),
            "patient": patient
        }
        for patient in os.listdir(input_dir)
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
    model = initialize_model(model_name)

    train_steps = len(train_ds) // BATCH_SIZE
    val_steps = len(val_ds) // BATCH_SIZE

    warp_layer = {
        "linear": Warp(mode="bilinear", padding_mode="border").to(DEVICE),
        "binary": Warp(mode="nearest", padding_mode="border").to(DEVICE)
    }
    criterion = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16, 32])
    regularization = None if model_name == "globalnet" else BendingEnergyLoss()
    if model_name == "globalnet":
        optimizer = Adam(model.parameters(), lr=INIT_LR, weight_decay=WD)
    else:
        optimizer = Adam(model.parameters(), lr=INIT_LR)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    dice_values = []
    best_dice = 0

    start_time = time()
    for e in tqdm(range(NUM_EPOCHS)):
        epoch_time = time()

        # Train and validate the model.
        train_loss = train(model, train_loader, criterion, regularization, optimizer, warp_layer)
        val_loss, val_dice_avg = validate(model, val_loader, criterion, regularization, warp_layer, dice_metric)

        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps

        # Print the model training and validation information.
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print(f"[INFO] Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Dice Index: {val_dice_avg}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "dice": val_dice_avg}, step=e+1)

        # Follow the epoch with the best dice and save the respective weights.
        dice_values.append(val_dice_avg)
        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            torch.save(model.state_dict(), f"{model_path}_best.pth")
            print(f"[INFO] saved model in epoch {e + 1} as new best model.")

        wandb.log({"best_dice": best_dice}, step=e+1)
        print(f"[INFO] Epoch time: {round((time() - epoch_time) / 60, 2)} minutes")

    print(f"[INFO] total time taken to train the model: {round(((time() - start_time) / 60) / 60, 2)} hours")
    wandb.finish()

    # Load saved model.
    model.load_state_dict(torch.load(f"{model_path}_best.pth"))

    # Calculate initial dice average.
    init_dice_avg = initial_dice(val_loader, dice_metric)
    print(f"[INFO] Average Initial Dice: {init_dice_avg}")

    # Inference the testing data.
    avg_dice, predictions = inference_model(model, val_loader, warp_layer, dice_metric)
    print(f"[INFO] Average Final Dice: {avg_dice}")

    # Save the predictions.
    save_predictions(predictions, output_dir)


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
