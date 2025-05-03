from torch.utils.data import Dataset
import os
import tifffile
from torchvision import transforms
import numpy as np
import torch

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images_names = os.listdir(img_dir)
        self.images_names = [name for name in self.images_names if name.endswith(('.tif', '.tiff'))]

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images_names[idx])
        image_name = self.images_names[idx].split(".")[0]
        mask_path = os.path.join(self.mask_dir, image_name + ".npy")

        #image = Image.open(image_path).convert("RGB")
        image = tifffile.imread(image_path)
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
