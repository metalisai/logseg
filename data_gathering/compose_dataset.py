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
    try:
        with open(jsonfile, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {jsonfile} not found. Skipping folder {folder}.")
        continue

    imgs = data['imgs']
    imgs.sort()

    if data['delete']:
        continue

    #print(f"{data['chosenDate'][0][5:]}")

    choice = 0
    if data['chosenDate'][0][5:] == "05-01":
        choice = 1
    elif data['chosenDate'][0][5:] == "09-15":
        choice = 2
    elif data['chosenDate'][0][5:] == "11-15":
        choice = 3

    img = imgs[choice]
    if str(data['year']) in img:
        img_path = os.path.join(folder_path, img)
        infolder = os.listdir(img_path)
        infolder = os.path.join(img_path, infolder[0])
        image_path = os.path.join(infolder, 'response.tiff')
        # copy image to output folder
        output_path = os.path.join(output_folder, f"{folder}.tiff")
        # copy
        os.system(f"cp {image_path} {output_path}")
