import os
import json

data_folder = 'sentinel_data_extra'

# make 'output' folder
output_folder = "output_extra"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

eraldised = os.listdir(data_folder)
for folder in eraldised:
    folder_path = os.path.join(data_folder, folder)
    jsonfile = os.path.join(folder_path, f"{folder}.json")
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    imgs = data['imgs']
    imgs.sort()

    for img in imgs:
        if str(data['year']) in img:
            img_path = os.path.join(folder_path, img)
            infolder = os.listdir(img_path)
            infolder = os.path.join(img_path, infolder[0])
            image_path = os.path.join(infolder, 'response.tiff')
            # copy image to output folder
            output_path = os.path.join(output_folder, f"{folder}.tiff")
            # copy
            os.system(f"cp {image_path} {output_path}")
