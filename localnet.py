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
from monai.transforms import Compose, LoadImaged, Resized
from monai.losses import DiceLoss, MultiScaleLoss, BendingEnergyLoss
from monai.metrics import DiceMetric

from registration import calculate_metrics
from utils import setup_parser, create_dir, validate_paths
from config.training import ARCHITECTURE, NUM_EPOCHS, INIT_LR, TR_BATCH_SIZE, VAL_BATCH_SIZE, DEVICE, PIN_MEMORY, \
    TEST_BATCH_SIZE

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
        tags=["localnet", "registration", "tumor_included_loss"],
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
def save_prediction(prediction, input_dir, output_dir):
    patient = list(prediction.keys())[0]
    print(f"Saving predictions: {patient}")

    patient_dir = create_dir(output_dir, patient)
    ct_volume_ref = sitk.ReadImage(f"{input_dir}/{patient}/spect_ct_volume.nii.gz")

    save_items = [
        "spect_ct_liver", "spect_ct_tumor", "mri_volume_pred",
        "mri_liver", "mri_tumor", "mri_liver_pred", "mri_tumor_pred",
        "mri_ddf_pred"
    ]
    for item in save_items:
        image = prediction[patient][item].cpu().numpy()[0, 0]
        image = np.swapaxes(image, 0, 2)
        image = sitk.GetImageFromArray(image)
        image.SetSpacing(ct_volume_ref.GetSpacing())
        image.SetOrigin(ct_volume_ref.GetOrigin())
        image.SetDirection(ct_volume_ref.GetDirection())
        sitk.WriteImage(image, f"{patient_dir}/{item}.nii.gz")


# Preprocessing and data augmentation.
def transforms():
    return Compose(
        [
            LoadImaged(
                keys=[
                    "fixed_image",
                    "fixed_unet3d_liver",
                    "fixed_liver",
                    "fixed_tumor",
                    "moving_image",
                    "moving_liver",
                    "moving_tumor",
                ],
                ensure_channel_first=True
            ),
            Resized(
                keys=[
                    "fixed_image",
                    "fixed_unet3d_liver",
                    "fixed_liver",
                    "fixed_tumor",
                    "moving_image",
                    "moving_liver",
                    "moving_tumor",
                ],
                mode=("trilinear", "nearest", "nearest", "nearest", "trilinear", "nearest", "nearest"),
                align_corners=(True, None, None, None, True, None, None),
                spatial_size=(256, 256, 224),
            ),
        ]
    )


# Forward the input to the model to get the prediction.
def forward(sample, model, warp_layer, inference=False):
    x_fixed = sample["fixed_image"].to(DEVICE)
    x_moving = sample["moving_image"].to(DEVICE)
    liver_moving = sample["moving_liver"].to(DEVICE)
    tumor_moving = sample["moving_liver"].to(DEVICE)

    # Predict DDF from fixed and moving images
    ddf = model(torch.cat((x_fixed, x_moving), dim=1))

    if inference:
        x_pred = warp_layer["linear"](x_moving, ddf)
        liver_pred = warp_layer["binary"](liver_moving, ddf)
        tumor_pred = warp_layer["binary"](tumor_moving, ddf)
        return ddf, x_pred, liver_pred, tumor_pred
    else:
        liver_pred_ln = warp_layer["linear"](liver_moving, ddf)
        liver_pred = warp_layer["binary"](liver_moving, ddf)
        tumor_pred = warp_layer["binary"](tumor_moving, ddf)
        return ddf, liver_pred_ln, liver_pred, tumor_pred


# Train the model.
def train(model, train_loader, criterion, regularization, optimizer, warp_layer, dice_metric):
    start_time = time()

    model.train()
    total_train_loss = 0

    timer = {"Loading": 0, "Forward": 0, "Loss": 0, "Backpropagation": 0, "Dice": 0}

    # loop over the training set.
    dice_liver, dice_tumor = dice_metric, dice_metric
    for batch_data in train_loader:
        load_time = time()

        # Predict DDF and moving label.
        ddf, liver_pred_ln, liver_pred, tumor_pred = forward(batch_data, model, warp_layer)
        fixed_unet3d_liver = batch_data["fixed_unet3d_liver"].to(DEVICE)
        fixed_liver = batch_data["fixed_liver"].to(DEVICE)
        fixed_tumor = batch_data["fixed_tumor"].to(DEVICE)
        liver_pred_ln[liver_pred_ln > 1] = 1
        fwd_calc = time()

        optimizer.zero_grad()

        # Calculate loss and backpropagation.
        train_loss = criterion(liver_pred_ln, fixed_unet3d_liver) + (0.5 * regularization(ddf))
        loss_calc = time()
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
        back_calc = time()

        # Calculate dice accuracy.
        dice_liver(y_pred=liver_pred, y=fixed_liver)
        dice_tumor(y_pred=tumor_pred, y=fixed_tumor)
        dice_calc = time()

        # Setup timer logs.
        timer["Loading"] += load_time - start_time
        timer["Forward"] += fwd_calc - load_time
        timer["Loss"] += loss_calc - fwd_calc
        timer["Backpropagation"] += back_calc - loss_calc
        timer["Dice"] += dice_calc - back_calc

        start_time = time()

    # Aggregate dice results.
    dice_liver_avg = dice_liver.aggregate().item()
    dice_tumor_avg = dice_tumor.aggregate().item()
    dice_liver.reset()
    dice_tumor.reset()

    print_time_logs(timer)

    return total_train_loss, [dice_liver_avg, dice_tumor_avg]


# Validate the model.
def validate(model, val_loader, criterion, regularization, warp_layer, dice_metric):
    start_time = time()

    model.eval()
    total_val_loss = 0

    timer = {"Loading": 0, "Forward": 0, "Loss": 0, "Dice": 0}

    # Loop over the validation set.
    with torch.no_grad():
        dice_liver, dice_tumor = dice_metric, dice_metric
        for val_data in val_loader:
            load_time = time()

            # Predict DDF and moving label.
            ddf, liver_pred_ln, liver_pred, tumor_pred = forward(val_data, model, warp_layer)
            fixed_unet3d_liver = val_data["fixed_unet3d_liver"].to(DEVICE)
            fixed_liver = val_data["fixed_liver"].to(DEVICE)
            fixed_tumor = val_data["fixed_tumor"].to(DEVICE)
            liver_pred_ln[liver_pred_ln > 1] = 1
            fwd_calc = time()

            # Calculate loss.
            val_loss = criterion(liver_pred_ln, fixed_unet3d_liver) + (0.5 * regularization(ddf))
            total_val_loss += val_loss.item()
            loss_calc = time()

            # Calculate dice accuracy.
            dice_liver(y_pred=liver_pred, y=fixed_liver)
            dice_tumor(y_pred=tumor_pred, y=fixed_tumor)
            dice_calc = time()

            timer["Loading"] += load_time - start_time
            timer["Forward"] += fwd_calc - load_time
            timer["Loss"] += loss_calc - fwd_calc
            timer["Dice"] += dice_calc - loss_calc

            start_time = time()

        # Aggregate dice results.
        dice_liver_avg = dice_liver.aggregate().item()
        dice_tumor_avg = dice_tumor.aggregate().item()
        dice_liver.reset()
        dice_tumor.reset()

    print_time_logs(timer)

    return total_val_loss, [dice_liver_avg, dice_tumor_avg]


# Inference the model.
def inference_model(model, test_loader, warp_layer, input_dir, output_dir):
    model.eval()

    print("Inferencing data...")
    # Loop over the validation set.
    with torch.no_grad():
        total_inference_time = 0
        for test_data in test_loader:
            start_time = time()

            # Predict moving image and moving label.
            ddf, x_pred, liver_pred, tumor_pred = forward(test_data, model, warp_layer, inference=True)

            inference_time = time() - start_time
            total_inference_time += inference_time
            print(f"{test_data['patient'][0]} inference time: {round(inference_time, 2)} sec.")

            save_prediction({
                test_data["patient"][0]: {
                    "spect_ct_liver": test_data["fixed_liver"].to(DEVICE),
                    "spect_ct_tumor": test_data["fixed_tumor"].to(DEVICE),
                    "mri_volume_pred": x_pred,
                    "mri_liver": test_data["moving_liver"].to(DEVICE),
                    "mri_liver_pred": liver_pred,
                    "mri_tumor": test_data["moving_tumor"].to(DEVICE),
                    "mri_tumor_pred": tumor_pred,
                    "mri_ddf_pred": ddf
                }
            }, input_dir, output_dir)

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
    output_dir = create_dir(dir_name, output_dir)

    # Initialize the model.
    model_dir = create_dir(output_dir, model_name)

    # Initialize wandb.
    init_wandb(fold)

    # Load dataset file paths.
    data = [
        {
            "fixed_image": os.path.join(input_dir, str(patient), "spect_ct_volume.nii.gz"),
            "fixed_unet3d_liver": os.path.join(input_dir, str(patient), "spect_ct_unet3d_liver.nii.gz"),
            "fixed_liver": os.path.join(input_dir, str(patient), "spect_ct_liver.nii.gz"),
            "fixed_tumor": os.path.join(input_dir, str(patient), "spect_ct_tumor.nii.gz"),
            "moving_image": os.path.join(input_dir, str(patient), "mri_volume.nii.gz"),
            "moving_liver": os.path.join(input_dir, str(patient), "mri_liver.nii.gz"),
            "moving_tumor": os.path.join(input_dir, str(patient), "mri_tumor.nii.gz"),
            "patient": patient
        }
        for patient in os.listdir(input_dir)
    ]

    # Create a k-fold cross-validation object
    k_folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = list(k_folds.split(data))[fold]
    train_files, val_files = [data[i] for i in train_idx], [data[i] for i in val_idx]
    test_files = val_files

    print("***Training Patients***")
    for item in train_files:
        print(item["patient"])

    print("***Validation Patients***")
    for item in val_files:
        print(item["patient"])

    print("***Testing Patients***")
    for item in test_files:
        print(item["patient"])

    # Cache the transforms of the datasets.
    train_ds = CacheDataset(data=train_files, transform=transforms(), cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_files, transform=transforms(), cache_rate=1.0, num_workers=0)
    test_ds = CacheDataset(data=test_files, transform=transforms(), cache_rate=1.0, num_workers=0)

    # Load the datasets.
    train_loader = DataLoader(train_ds, batch_size=TR_BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False, num_workers=0)

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

    best_liver_dice = 0
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
            "val_liver_dice": val_dice_avg[0], "train_liver_dice": train_dice_avg[0],
            "val_tumor_dice": val_dice_avg[1], "train_tumor_dice": train_dice_avg[1]
        }, step=e + 1)

        # Follow the epoch with the best dice and save the respective weights.
        if val_dice_avg[0] > best_liver_dice:
            best_liver_dice = val_dice_avg[0]
            torch.save(model.state_dict(), f"{model_dir}/fold_{fold}_best_model.pth")
            print(f"[INFO] saved model in epoch {e + 1} as new best model.")

        wandb.log({"best_liver_dice": best_liver_dice}, step=e + 1)
        print(f"[INFO] Epoch time: {round((time() - epoch_start_time) / 60, 2)} minutes")

    print(f"\n[INFO] total time taken to train the model: {round(((time() - start_time) / 60) / 60, 2)} hours")
    wandb.finish()

    # Load saved model.
    model.load_state_dict(torch.load(f"{model_dir}/fold_{fold}_best_model.pth"))

    # Inference the testing data and save the predictions.
    fold_data_output = create_dir(model_dir, f"fold_{fold}_raw")
    inference_model(model, test_loader, warp_layer, input_dir, fold_data_output)

    # Calculate metrics.
    patient_list = [item["patient"] for item in test_files]
    for patient in patient_list:
        patient_dir = os.path.join(fold_data_output, patient)
        spect_ct_liver = os.path.join(str(patient_dir), "spect_ct_liver.nii.gz")
        spect_ct_tumor = os.path.join(str(patient_dir), "spect_ct_tumor.nii.gz")
        mri_liver = os.path.join(str(patient_dir), "mri_liver.nii.gz")
        mri_liver_pred = os.path.join(str(patient_dir), "mri_liver_pred.nii.gz")
        mri_tumor = os.path.join(str(patient_dir), "mri_tumor.nii.gz")
        mri_tumor_pred = os.path.join(str(patient_dir), "mri_tumor_pred.nii.gz")

        results = {
            "liver": {
                "Initial": calculate_metrics(spect_ct_liver, mri_liver),
                "LocalNet": calculate_metrics(spect_ct_liver, mri_liver_pred)
            },
            "tumor": {
                "Initial": calculate_metrics(spect_ct_tumor, mri_tumor),
                "LocalNet": calculate_metrics(spect_ct_tumor, mri_tumor_pred)
            }
        }

        with open(f"{patient_dir}/evaluation.json", "w") as fp:
            json.dump(results, fp)

        print(f"Patient {patient} Liver Dice: {results['liver']['Initial']} ---> {results['liver']['LocalNet']}")
        print(f"Patient {patient} Liver Dice: {results['tumor']['Initial']} ---> {results['tumor']['LocalNet']}\n")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
