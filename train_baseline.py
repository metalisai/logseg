from torchgeo.models import ResNet18_Weights, ResNet50_Weights
from torchgeo.trainers import SemanticSegmentationTask
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
#from PIL import Image
import tifffile
import os
import numpy as np
import torch
import torchmetrics

img_dir = "./images_falsecolor"
mask_dir = "./masks"
channels = 13 if "allbands" in img_dir else 3
name = "falsecolor"

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Load your dataset here

        self.images_names = os.listdir(img_dir)
        self.images_names = [name for name in self.images_names if name.endswith(('.tif', '.tiff'))]

    def __len__(self):
        # Return the number of samples in your dataset
        return len(self.images_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images_names[idx])
        image_name = self.images_names[idx].split(".")[0]
        mask_path = os.path.join(self.mask_dir, image_name + ".npy")

        #image = Image.open(image_path).convert("RGB")
        image = tifffile.imread(image_path)
        #print(image.shape)
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
        else:
            print(f"Warning: Mask not found for {image_path}. Using empty mask.")
            #mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform:
            image = self.transform(image)

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
        }

transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(180),
    #v2.RandomCrop(size=(128, 128)),
    #v2.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
])

train_img_dir = os.path.join(img_dir, "train")
train_mask_dir = os.path.join(mask_dir, "train")
val_img_dir = os.path.join(img_dir, "valid")
val_mask_dir = os.path.join(mask_dir, "valid")
test_img_dir = os.path.join(img_dir, "test")
test_mask_dir = os.path.join(mask_dir, "test")

train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, transform=None)
test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transform=None)

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

weights = ResNet50_Weights.SENTINEL2_ALL_MOCO if channels == 13 else ResNet50_Weights.SENTINEL2_RGB_MOCO
#weights = ResNet18_Weights.SENTINEL2_RGB_MOCO

task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet50",
    weights=weights,
    in_channels=channels,
    task="binary",
    loss="bce",
)

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

trainer.test(
    task,
    dataloaders=test_dataloader,
)
    
trainer.fit(
    task,
    train_dataloader,
    val_dataloader,
)
trainer.test(
    dataloaders=test_dataloader,
)

import matplotlib.pyplot as plt

trainer.model.eval()  # Set the model to evaluation mode

# Get a batch of data from the validation dataloader
data_iter = iter(val_dataloader)
batch = next(data_iter)
images = batch["image"]
masks = batch["mask"]

# Pass the images through the trained model to get predictions
with torch.no_grad():
    logits = trainer.model(images)
    preds = torch.sigmoid(logits) > 0.5  # Binarize predictions for binary segmentation

# For visualization, we will plot images, masks, and predictions for a few samples
num_samples = 4  # Number of samples to display

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 12))

for i in range(num_samples):
    # Convert the tensors to numpy arrays for plotting
    img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format for display
    mask = masks[i].cpu().numpy()  # Ground truth mask
    pred = preds[i].cpu().numpy()  # Predicted mask

    # Plot the image, ground truth mask, and prediction
    img = img[..., 1:4] if channels == 13 else img  # Take only the first 3 channels for RGB
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image {i+1}")
    axes[i, 0].axis('off')

    mask = np.squeeze(mask)  # Remove channel dimension if present
    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 1].set_title(f"Ground Truth {i+1}")
    axes[i, 1].axis('off')

    pred = np.squeeze(pred)  # Remove channel dimension if present
    axes[i, 2].imshow(pred, cmap='gray')
    axes[i, 2].set_title(f"Prediction {i+1}")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
