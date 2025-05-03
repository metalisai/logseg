from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import os
import cv2

anns = "./extra/instances_default.json"
mask_dir = "./masks224/"
img_dir = "./images_falsecolor_224"

coco = COCO(anns)

rescale = 224

train_imgs = os.listdir(os.path.join(img_dir, "train"))
test_imgs = os.listdir(os.path.join(img_dir, "test"))
valid_imgs = os.listdir(os.path.join(img_dir, "valid"))

filesplts = {}
for img in train_imgs:
    filesplts[img.split('.')[0]] = "train"
for img in test_imgs:
    filesplts[img.split('.')[0]] = "test"
for img in valid_imgs:
    filesplts[img.split('.')[0]] = "valid"

for img_id in coco.getImgIds():
    img_info = coco.imgs[img_id]
    img_name = img_info['file_name']
    img_base = img_name.split('.')[0]
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    for ann in anns:
        rle = maskUtils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
        #rle = maskUtils.merge(rle)
        m = maskUtils.decode(rle)
        mask = np.maximum(mask, m)

    if rescale is not None:
        mask = cv2.resize(mask, (rescale, rescale), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8)
    print(f"Mask shape: {mask.shape}")
    if img_base not in filesplts:
        print(f"Warning: {img_base} not found in filesplts.")
        continue
    outdir = os.path.join(mask_dir, filesplts[img_base])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mask_path = os.path.join(outdir, str(img_base) + ".npy")
    np.save(mask_path, mask)
