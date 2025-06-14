from torchgeo.models import ResNet18_Weights, ResNet50_Weights
from torchgeo.trainers import SemanticSegmentationTask
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
#from PIL import Image
import os
import numpy as np
import torch
import torchmetrics
from dataloader import SegmentationDataset

img_dir = "./images_falsecolor"
mask_dir = "./masks"
channels = 13 if "allbands" in img_dir else 3
name = "falsecolor"

transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(180),
])

train_img_dir = os.path.join(img_dir, "train")
train_mask_dir = os.path.join(mask_dir, "train")
val_img_dir = os.path.join(img_dir, "valid")
val_mask_dir = os.path.join(mask_dir, "valid")
test_img_dir = os.path.join(img_dir, "test")
test_mask_dir = os.path.join(mask_dir, "test")

# Create datasets
train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, transform=None)
test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transform=None)

# Create data loaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
)

# Use ResNet50 as the backbone
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO if channels == 13 else ResNet50_Weights.SENTINEL2_RGB_MOCO

# Create the model
task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet50",
    weights=weights,
    in_channels=channels,
    task="binary",
    loss="bce",
)

# Log to TensorBoard
logger = TensorBoardLogger("runs", name=name)
trainer = Trainer(
    max_epochs=40,
    accelerator="gpu",
    log_every_n_steps=1,
    logger=logger, 
    #val_check_interval=1,
    check_val_every_n_epoch=1,
    #enable_checkpointing=True,
)

# Test the model before training (to see that it really is random)
trainer.test(
    task,
    dataloaders=test_dataloader,
)
    
# Train the model
trainer.fit(
    task,
    train_dataloader,
    val_dataloader,
)
    
# Test the model after training
trainer.test(
    dataloaders=test_dataloader,
)

import matplotlib.pyplot as plt

trainer.model.eval()

# Visualize some predictions

data_iter = iter(val_dataloader)
batch = next(data_iter)
images = batch["image"]
masks = batch["mask"]

with torch.no_grad():
    logits = trainer.model(images)
    preds = torch.sigmoid(logits) > 0.5

num_samples = 4

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 12))

for i in range(num_samples):
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    mask = masks[i].cpu().numpy()
    pred = preds[i].cpu().numpy()

    img = img[..., 1:4] if channels == 13 else img
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image {i+1}")
    axes[i, 0].axis('off')

    mask = np.squeeze(mask)
    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 1].set_title(f"Ground Truth {i+1}")
    axes[i, 1].axis('off')

    pred = np.squeeze(pred)
    axes[i, 2].imshow(pred, cmap='gray')
    axes[i, 2].set_title(f"Prediction {i+1}")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
