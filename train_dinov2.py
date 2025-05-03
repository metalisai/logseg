import torch
import torch.nn.functional as F
from dataloader import SegmentationDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os
import matplotlib.pyplot as plt
import numpy as np

img_dir = "./images_falsecolor_224"
mask_dir = "./masks224"

model_dtype = torch.bfloat16

train_img_dir = os.path.join(img_dir, "train")
train_mask_dir = os.path.join(mask_dir, "train")
val_img_dir = os.path.join(img_dir, "valid")
val_mask_dir = os.path.join(mask_dir, "valid")
test_img_dir = os.path.join(img_dir, "test")
test_mask_dir = os.path.join(mask_dir, "test")

transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(180),
])

train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, transform=None)
test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transform=None)

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

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

#print(dinov2)
dinov2 = dinov2.to(device="cuda", dtype=model_dtype)
dinov2.eval()

data_iter = iter(val_dataloader)
batch = next(data_iter)
images = batch["image"].to(device="cuda", dtype=model_dtype)

result = dinov2(images)
print(result.shape)

outputs = dinov2.forward_features(images)
patch_embeddings = outputs["x_norm_patchtokens"]
print(f"keys: {outputs.keys()}")
print(patch_embeddings.shape)

class MySegmentationModel(torch.nn.Module):
    def __init__(self, dinov2_model):
        super(MySegmentationModel, self).__init__()
        self.dinov2_model = dinov2_model
        self.linear1 = torch.nn.Conv2d(
            in_channels=dinov2_model.embed_dim,
            out_channels=196,
            kernel_size=1
        )
        self.linear2 = torch.nn.Conv2d(
            in_channels=196,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        features = self.dinov2_model.forward_features(x)
        B, P, C = features["x_norm_patchtokens"].shape
        S = int(P**0.5)
        x = features["x_norm_patchtokens"].reshape(B, S, S, C)
        x = x.permute(0, 3, 1, 2)
        x = self.linear1(x)
        x = F.silu(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.linear2(x)
        return x

myModel = MySegmentationModel(dinov2)
myModel = myModel.to(device="cuda", dtype=model_dtype)
outputs = myModel(images)
print(f"Out {outputs.shape}")

num_epochs = 20
optimizer = torch.optim.Adam(myModel.parameters(), lr=5e-5)
criterion = torch.nn.BCEWithLogitsLoss()

myModel.train()

def compute_iou(preds, masks, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    masks = masks.float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks - preds * masks).sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

for epoch in range(num_epochs):
    myModel.train()
    for batch in train_dataloader:
        images = batch["image"].to(device="cuda", dtype=model_dtype)
        masks = batch["mask"].float().to(device="cuda", dtype=model_dtype).unsqueeze(1)

        optimizer.zero_grad()
        outputs = myModel(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    myModel.eval()
    iou = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to(device="cuda", dtype=model_dtype)
            masks = batch["mask"].float().to(device="cuda", dtype=model_dtype).unsqueeze(1)

            outputs = myModel(images)
            iou += compute_iou(outputs, masks)
    iou /= len(val_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation IoU: {iou}")

# Eval on test set
myModel.eval()
iou = 0.0
with torch.no_grad():
    for batch in test_dataloader:
        images = batch["image"].to(device="cuda", dtype=model_dtype)
        masks = batch["mask"].float().to(device="cuda", dtype=model_dtype).unsqueeze(1)

        outputs = myModel(images)
        iou += compute_iou(outputs, masks)
    iou /= len(test_dataloader)
    print(f"Test IoU: {iou}")

# Show some examples

data_iter = iter(val_dataloader)
batch = next(data_iter)
images = batch["image"].to(device="cuda", dtype=model_dtype)
masks = batch["mask"]

with torch.no_grad():
    logits = myModel(images)
    preds = torch.sigmoid(logits) > 0.5

num_samples = 4

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 12))

for i in range(num_samples):
    img = images[i].float().cpu().numpy().transpose(1, 2, 0)
    mask = masks[i].float().cpu().numpy()
    pred = preds[i].float().cpu().numpy()

    #img = img[..., 1:4] if channels == 13 else img
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
