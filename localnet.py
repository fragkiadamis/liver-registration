import json
import os
import random
from time import time

import torch
from torch.optim import Adam
from sklearn.model_selection import KFold

import wandb
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import Compose, LoadImaged
from monai.losses import DiceLoss, MultiScaleLoss, BendingEnergyLoss
from monai.metrics import DiceMetric

from registration import calculate_metrics
from utils import setup_parser, create_dir, validate_paths
from config.training import ARCHITECTURE, NUM_EPOCHS, INIT_LR, TR_BATCH_SIZE, VAL_BATCH_SIZE, DEVICE, PIN_MEMORY

print_config()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Initialize wandb.
def init_wandb(fold):
    wandb.init(
        project="liver-registration",
        name=f"fold_{fold}",
        tags=["localnet", "registration"],
        config={
            "architecture": ARCHITECTURE,
            "epochs": NUM_EPOCHS,
            "learning_rate": INIT_LR,
            "batch_size": TR_BATCH_SIZE
        }
    )


# Print the time logs.
def print_time_logs(timer):
    for key, value in timer.items():
        print(f"[INFO] {key} Time: {round(value / 60, 2)} minutes.")


# Extract the images from the torches and save them.
def save_prediction(prediction, input_dir):
    patient = list(prediction.keys())[0]
    print(f"Saving predictions: {patient}")

    patient_dir = os.path.join(input_dir, patient)
    img_ref = sitk.ReadImage(f"./data/localnet/{patient}/ct_volume.nii.gz")

    image = prediction[patient]["image_tensor"].cpu().numpy()[0, 0]
    image = np.swapaxes(image, 0, 2)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(img_ref.GetSpacing())
    image.SetOrigin(img_ref.GetOrigin())
    image.SetDirection(img_ref.GetDirection())

    label = prediction[patient]["label_tensor"].cpu().numpy()[0, 0]
    label = np.swapaxes(label, 0, 2)
    label = sitk.GetImageFromArray(label)
    label.SetSpacing(img_ref.GetSpacing())
    label.SetOrigin(img_ref.GetOrigin())
    label.SetDirection(img_ref.GetDirection())

    ddf = prediction[patient]["ddf"].cpu().numpy()[0, 0]
    ddf = np.swapaxes(ddf, 0, 2)
    ddf = sitk.GetImageFromArray(ddf)
    ddf.SetSpacing(img_ref.GetSpacing())
    ddf.SetOrigin(img_ref.GetOrigin())
    ddf.SetDirection(img_ref.GetDirection())

    sitk.WriteImage(image, f"{patient_dir}/mri_volume_pred.nii.gz")
    sitk.WriteImage(label, f"{patient_dir}/mri_liver_pred.nii.gz")
    sitk.WriteImage(label, f"{patient_dir}/ddf_pred.nii.gz")


# Preprocessing and data augmentation.
def transforms():
    return Compose(
        [
            LoadImaged(
                keys=[
                    "fixed_image",
                    "fixed_unet3d_label",
                    "fixed_gt_label",
                    "moving_image",
                    "moving_label"
                ],
                ensure_channel_first=True
            ),
        ]
    )


# Forward the input to the model to get the prediction.
def forward(sample, model, warp_layer, inference=False):
    x_fixed = sample["fixed_image"].to(DEVICE)
    x_moving = sample["moving_image"].to(DEVICE)
    y_moving = sample["moving_label"].to(DEVICE)

    # Predict DDF from fixed and moving images
    ddf = model(torch.cat((x_fixed, x_moving), dim=1))

    if inference:
        x_pred = warp_layer["linear"](x_moving, ddf)
        y_pred = warp_layer["binary"](y_moving, ddf)
        return ddf, x_pred, y_pred
    else:
        y_pred_ln = warp_layer["linear"](y_moving, ddf)
        y_pred_bn = warp_layer["binary"](y_moving, ddf)
        return ddf, y_pred_ln, y_pred_bn


# Train the model.
def train(model, train_loader, criterion, regularization, optimizer, warp_layer, dice_metric):
    start_time = time()

    model.train()
    total_train_loss = 0

    timer = {"Loading": 0, "Forward": 0, "Loss": 0, "Backpropagation": 0, "Dice": 0}

    # loop over the training set.
    for batch_data in train_loader:
        load_time = time()

        # Predict DDF and moving label.
        ddf, y_pred_ln, y_pred_bn = forward(batch_data, model, warp_layer)
        fixed_unet3d_label = batch_data["fixed_unet3d_label"].to(DEVICE)
        y_pred_ln[y_pred_ln > 1] = 1
        fwd_calc = time()

        optimizer.zero_grad()

        # Calculate loss and backpropagation.
        train_loss = criterion(y_pred_ln, fixed_unet3d_label) + (0.5 * regularization(ddf))
        loss_calc = time()
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
        back_calc = time()

        # Calculate dice accuracy.
        fixed_gt_label = batch_data["fixed_gt_label"].to(DEVICE)
        dice_metric(y_pred=y_pred_bn, y=fixed_gt_label)
        dice_calc = time()

        # Setup timer logs.
        timer["Loading"] += load_time - start_time
        timer["Forward"] += fwd_calc - load_time
        timer["Loss"] += loss_calc - fwd_calc
        timer["Backpropagation"] += back_calc - loss_calc
        timer["Dice"] += dice_calc - back_calc

        start_time = time()

    # Aggregate dice results.
    dice_avg = dice_metric.aggregate().item()
    dice_metric.reset()

    print_time_logs(timer)

    return total_train_loss, dice_avg


# Validate the model.
def validate(model, val_loader, criterion, regularization, warp_layer, dice_metric):
    start_time = time()

    model.eval()
    total_val_loss = 0

    timer = {"Loading": 0, "Forward": 0, "Loss": 0, "Dice": 0}

    # Loop over the validation set.
    with torch.no_grad():
        for val_data in val_loader:
            load_time = time()

            # Predict DDF and moving label.
            ddf, y_pred_ln, y_pred_bn = forward(val_data, model, warp_layer)
            fixed_unet3d_label = val_data["fixed_unet3d_label"].to(DEVICE)
            y_pred_ln[y_pred_ln > 1] = 1
            fwd_calc = time()

            # Calculate loss.
            val_loss = criterion(y_pred_ln, fixed_unet3d_label) + (0.5 * regularization(ddf))
            total_val_loss += val_loss.item()
            loss_calc = time()

            # Calculate dice accuracy.
            fixed_gt_label = val_data["fixed_gt_label"].to(DEVICE)
            dice_metric(y_pred=y_pred_bn, y=fixed_gt_label)
            dice_calc = time()

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
def inference_model(model, test_loader, warp_layer, output_dir):
    model.eval()

    print("Inferencing data...")
    # Loop over the validation set.
    with torch.no_grad():
        total_inference_time = 0
        for test_data in test_loader:
            start_time = time()

            # Predict moving image and moving label.
            ddf, x_pred, y_pred = forward(test_data, model, warp_layer, inference=True)

            inference_time = time() - start_time
            total_inference_time += inference_time
            print(f"{test_data['patient'][0]} inference time: {round(inference_time, 2)} sec.")

            save_prediction({
                test_data["patient"][0]: {
                    "image_tensor": x_pred,
                    "label_tensor": y_pred,
                    "ddf": ddf
                }
            }, output_dir)

        avg_inference_time = total_inference_time / len(test_loader)
        print(f"Average inference time: {round(avg_inference_time, 2)} sec.")


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/localnet_parser.json")
    input_dir = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)
    model_name = args.n
    fold = int(args.f)

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_dir, output_dir)
    create_dir(dir_name, output_dir)

    # Initialize the model.
    model_dir = create_dir(output_dir, model_name)

    # Initialize wandb.
    init_wandb(fold)

    # Load dataset file paths.
    data = [
        {
            "fixed_image": os.path.join(input_dir, str(patient), "ct_volume.nii.gz"),
            "fixed_unet3d_label": os.path.join(input_dir, str(patient), "ct_unet3d_liver.nii.gz"),
            "fixed_gt_label": os.path.join(input_dir, str(patient), "ct_liver.nii.gz"),
            "moving_image": os.path.join(input_dir, str(patient), "mri_volume.nii.gz"),
            "moving_label": os.path.join(input_dir, str(patient), "mri_liver.nii.gz"),
            "patient": patient
        }
        for patient in os.listdir(input_dir)
    ]

    # Create a k-fold cross-validation object
    k_folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = list(k_folds.split(data))[fold]
    train_files, val_files = [data[i] for i in train_idx], [data[i] for i in val_idx]

    print("***Training Patients***")
    for item in train_files:
        print(item["patient"])

    print("***Validation Patients***")
    for item in val_files:
        print(item["patient"])

    # Cache the transforms of the datasets.
    train_ds = CacheDataset(data=train_files, transform=transforms(), cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_files, transform=transforms(), cache_rate=1.0, num_workers=0)

    # Load the datasets.
    train_loader = DataLoader(train_ds, batch_size=TR_BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False, num_workers=0)

    # Create LocalNet, losses and optimizer.
    model = LocalNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        num_channel_initial=16,
        extract_levels=(2,),
        out_activation=None,
        out_kernel_initializer="zeros",
    ).to(DEVICE)

    train_steps = len(train_ds) // TR_BATCH_SIZE
    val_steps = len(val_ds) // VAL_BATCH_SIZE

    warp_layer = {
        "linear": Warp(mode="bilinear", padding_mode="border").to(DEVICE),
        "binary": Warp(mode="nearest", padding_mode="border").to(DEVICE)
    }
    criterion = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16, 32])
    regularization = BendingEnergyLoss()
    optimizer = Adam(model.parameters(), lr=INIT_LR)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    dice_values = []
    best_dice = 0

    start_time = time()
    for e in tqdm(range(NUM_EPOCHS)):
        epoch_start_time = time()

        # Train and validate the model.
        print("\n***TRAINING***")
        train_loss, train_dice_avg = train(model, train_loader, criterion,
                                           regularization, optimizer, warp_layer, dice_metric)
        print(f"[INFO] Training time: {round((time() - epoch_start_time) / 60, 2)} minutes")

        print("***VALIDATION***")
        val_start_time = time()
        val_loss, val_dice_avg = validate(model, val_loader, criterion, regularization, warp_layer, dice_metric)
        print(f"[INFO] Validation time: {round((time() - val_start_time) / 60, 2)} minutes")

        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps

        # Print the model training and validation information.
        print("***EPOCH***")
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print(f"[INFO] Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Dice Index: {val_dice_avg}")
        wandb.log({
            "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "val_dice_accuracy": val_dice_avg, "train_dice_accuracy": train_dice_avg
        }, step=e + 1)

        # Follow the epoch with the best dice and save the respective weights.
        dice_values.append(val_dice_avg)
        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            torch.save(model.state_dict(), f"{model_dir}/fold_{fold}_best_model.pth")
            print(f"[INFO] saved model in epoch {e + 1} as new best model.")

        wandb.log({"best_dice": best_dice}, step=e + 1)
        print(f"[INFO] Epoch time: {round((time() - epoch_start_time) / 60, 2)} minutes")

    print(f"\n[INFO] total time taken to train the model: {round(((time() - start_time) / 60) / 60, 2)} hours")
    wandb.finish()

    # Load saved model.
    model.load_state_dict(torch.load(f"{model_dir}/fold_{fold}_best_model.pth"))

    # Inference the testing data and save the predictions.
    inference_model(model, val_loader, warp_layer, input_dir)

    # Calculate metrics.
    patient_list = [item["patient"] for item in val_files]
    for patient in patient_list:
        patient_dir = os.path.join(input_dir, patient)
        ct_gt_liver = os.path.join(patient_dir, "ct_liver.nii.gz")
        mri_liver = os.path.join(patient_dir, "mri_liver.nii.gz")
        mri_liver_pred = os.path.join(patient_dir, "mri_liver_pred.nii.gz")

        results = {
            "liver": {
                "Initial": calculate_metrics(ct_gt_liver, mri_liver),
                "LocalNet": calculate_metrics(ct_gt_liver, mri_liver_pred)
            }
        }

        with open(f"{patient_dir}/evaluation.json", "w") as fp:
            json.dump(results, fp)

        print(f"Patient {patient} Dice: {results['liver']['Initial']} ---> {results['liver']['LocalNet']}")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
