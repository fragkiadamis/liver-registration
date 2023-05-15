import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss
import monai
from monai.apps import download_url
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
)
from monai.utils import set_determinism, first

print_config()
data_dir = "data/dl"

data_dicts = [
    {
        "fixed_image": os.path.join(data_dir, f"images/{patient}_fixed.nii.gz"),
        "moving_image": os.path.join(data_dir, f"images/{patient}_moving.nii.gz"),
        "fixed_label": os.path.join(data_dir, f"labels/{patient}_fixed.nii.gz"),
        "moving_label": os.path.join(data_dir, f"labels/{patient}_moving.nii.gz"),
    }
    for patient in os.listdir("data/mri_spect_nii_iso")
]

train_files, val_files = data_dicts[:23], data_dicts[23:]

set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["fixed_image", "moving_image"],
            a_min=-285,
            a_max=3770,
            b_min=-0.5,
            b_max=0.5,
            clip=True,
        ),
        Resized(
            keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
            mode=("trilinear", "trilinear", "nearest", "nearest"),
            align_corners=(True, True, None, None),
            spatial_size=(96, 96, 104),
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["fixed_image", "moving_image"],
            a_min=-285,
            a_max=3770,
            b_min=-0.5,
            b_max=0.5,
            clip=True,
        ),
        Resized(
            keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
            mode=("trilinear", "trilinear", "nearest", "nearest"),
            align_corners=(True, True, None, None),
            spatial_size=(96, 96, 104),
        ),
    ]
)

check_ds = Dataset(data=train_files, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
fixed_image = check_data["fixed_image"][0][0].permute(1, 0, 2)
fixed_label = check_data["fixed_label"][0][0].permute(1, 0, 2)
moving_image = check_data["moving_image"][0][0].permute(1, 0, 2)
moving_label = check_data["moving_label"][0][0].permute(1, 0, 2)

print(f"moving_image shape: {moving_image.shape}, " f"moving_label shape: {moving_label.shape}")
print(f"fixed_image shape: {fixed_image.shape}, " f"fixed_label shape: {fixed_label.shape}")

# plot the slice [:, :, 50]
plt.figure("check", (12, 6))
plt.subplot(1, 4, 1)
plt.title("moving_image")
plt.imshow(moving_image[:, :, 50], cmap="gray")
plt.subplot(1, 4, 2)
plt.title("moving_label")
plt.imshow(moving_label[:, :, 50])
plt.subplot(1, 4, 3)
plt.title("fixed_image")
plt.imshow(fixed_image[:, :, 50], cmap="gray")
plt.subplot(1, 4, 4)
plt.title("fixed_label")
plt.imshow(fixed_label[:, :, 50])

plt.show()
plt.show()

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# standard PyTorch program style: create LocalNet, losses and optimizer
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
warp_layer = Warp().to(device)
image_loss = MSELoss()
label_loss = DiceLoss()
# label_loss = MultiScaleLoss(label_loss, scales=[0, 1, 2, 4, 8, 16])
regularization = BendingEnergyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)


def forward(batch_data, model):
    fixed_image = batch_data["fixed_image"].to(device)
    moving_image = batch_data["moving_image"].to(device)
    moving_label = batch_data["moving_label"].to(device)

    # predict DDF through LocalNet
    ddf = model(torch.cat((moving_image, fixed_image), dim=1))

    # warp moving image and label with the predicted ddf
    pred_image = warp_layer(moving_image, ddf)
    pred_label = warp_layer(moving_label, ddf)

    return ddf, pred_image, pred_label


max_epochs = 5
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    if (epoch + 1) % val_interval == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_ddf, val_pred_image, val_pred_label = forward(val_data, model)
                val_pred_label[val_pred_label > 1] = 1

                val_fixed_image = val_data["fixed_image"].to(device)
                val_fixed_label = val_data["fixed_label"].to(device)
                dice_metric(y_pred=val_pred_label, y=val_fixed_label)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(data_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} "
                f"current mean dice: {metric:.4f}\n"
                f"best mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()

        ddf, pred_image, pred_label = forward(batch_data, model)
        pred_label[pred_label > 1] = 1

        fixed_image = batch_data["fixed_image"].to(device)
        fixed_label = batch_data["fixed_label"].to(device)
        loss = (
            image_loss(pred_image, fixed_image) + 100 * label_loss(pred_label, fixed_label) + 10 * regularization(ddf)
        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

print(f"train completed, " f"best_metric: {best_metric:.4f}  " f"at epoch: {best_metric_epoch}")

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()

resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/pair_lung_ct.pth"
dst = f"{data_dir}/pretrained_weight.pth"
download_url(resource, dst)
model.load_state_dict(torch.load(dst))

model.eval()
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        if i > 2:
            break
        val_ddf, val_pred_image, val_pred_label = forward(val_data, model)
        val_pred_image = val_pred_image.cpu().numpy()[0, 0].transpose((1, 0, 2))
        val_pred_label = val_pred_label.cpu().numpy()[0, 0].transpose((1, 0, 2))
        val_moving_image = val_data["moving_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
        val_moving_label = val_data["moving_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))
        val_fixed_image = val_data["fixed_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
        val_fixed_label = val_data["fixed_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))

        for depth in range(10):
            depth = depth * 10
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 6, 1)
            plt.title(f"moving_image {i} d={depth}")
            plt.imshow(val_moving_image[:, :, depth], cmap="gray")
            plt.subplot(1, 6, 2)
            plt.title(f"moving_label {i} d={depth}")
            plt.imshow(val_moving_label[:, :, depth])
            plt.subplot(1, 6, 3)
            plt.title(f"fixed_image {i} d={depth}")
            plt.imshow(val_fixed_image[:, :, depth], cmap="gray")
            plt.subplot(1, 6, 4)
            plt.title(f"fixed_label {i} d={depth}")
            plt.imshow(val_fixed_label[:, :, depth])
            plt.subplot(1, 6, 5)
            plt.title(f"pred_image {i} d={depth}")
            plt.imshow(val_pred_image[:, :, depth], cmap="gray")
            plt.subplot(1, 6, 6)
            plt.title(f"pred_label {i} d={depth}")
            plt.imshow(val_pred_label[:, :, depth])
            plt.show()
