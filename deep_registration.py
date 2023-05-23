import os
import matplotlib.pyplot as plt

import torch
import wandb

from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import Compose, LoadImaged, Resized
from monai.utils import set_determinism

from utils import setup_parser, create_dir, validate_paths

print_config()

wandb.init(
    # set the wandb project where this run will be logged
    project="liver-registration",

    # track hyperparameters and run metadata
    config={
        "architecture": "LocalNet",
        "epochs": 5,
    }
)


def preprocessing():
    return Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("trilinear", "trilinear", "nearest", "nearest"),
                align_corners=(True, True, None, None),
                spatial_size=(249, 249, 201),
            ),
        ]
    )


def forward(batch_data, model, warp_layer, device):
    fixed_image = batch_data["fixed_image"].to(device)
    moving_image = batch_data["moving_image"].to(device)
    moving_label = batch_data["moving_label"].to(device)

    # predict DDF through LocalNet
    ddf = model(torch.cat((moving_image, fixed_image), dim=1))

    # warp moving image and label with the predicted ddf
    pred_label = warp_layer["label"](moving_label, ddf)

    return ddf, pred_label


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/parser/dl_registration_parser.json")
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
    train_files, val_files = data_dicts[:5], data_dicts[5:6]

    set_determinism(seed=0)

    # Cache the transforms of the datasets.
    train_ds = CacheDataset(data=train_files, transform=preprocessing(), cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_files, transform=preprocessing(), cache_rate=1.0, num_workers=0)

    # Load the datasets.
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    # Create LocalNet, losses and optimizer.
    device = torch.device("cuda:0")
    model = LocalNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        num_channel_initial=32,
        extract_levels=(3,),
        out_activation=None,
        out_kernel_initializer="zeros",
    ).to(device)

    warp_layer = {"image": Warp().to(device), "label": Warp(mode="nearest").to(device)}
    label_loss = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16])
    regularization = BendingEnergyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    max_epochs, val_interval = 5, 1
    best_metric, best_metric_epoch = -1, -1
    epoch_loss_values, metric_values = [], []

    for epoch in range(max_epochs):
        metric = 0
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_ddf, val_pred_label = forward(val_data, model, warp_layer, device)
                    val_pred_label[val_pred_label > 1] = 1

                    val_fixed_label = val_data["fixed_label"].to(device)
                    dice_metric(y_pred=val_pred_label, y=val_fixed_label)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(output_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                # print(
                #     f"current epoch: {epoch + 1} "
                #     f"current mean dice: {metric:.4f}\n"
                #     f"best mean dice: {best_metric:.4f} "
                #     f"at epoch: {best_metric_epoch}"
                # )
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()

            ddf, pred_label = forward(val_data, model, warp_layer, device)
            pred_label[pred_label > 1] = 1

            fixed_label = batch_data["fixed_label"].to(device)
            loss = (100 * label_loss(pred_label, fixed_label) + 10 * regularization(ddf))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # log metrics to wandb
            wandb.log({"dice": metric, "loss": loss})

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    print(f"train completed, " f"best_metric: {best_metric:.4f}  " f"at epoch: {best_metric_epoch}")


# Use this file as a script and run it.
if __name__ == '__main__':
    main()
