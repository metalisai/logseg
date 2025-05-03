from rasterio.transform import xy
import matplotlib.pyplot as plt
import rasterio
import os
import json
import requests
from PIL import Image
from io import BytesIO
import argparse

output_dir = 'output_extra'
files = os.listdir(output_dir)
sdata_dir = 'sentinel_data_extra'

def fetch_wms_image(file):
    path = os.path.join(output_dir, file)
    with rasterio.open(path) as src:
        print(f"Opening {path}")

        width = src.width
        height = src.height
        transform = src.transform

        # (row, col) to (x, y) using rasterio.transform.xy
        topleft = xy(transform, 0, 0)
        topright = xy(transform, 0, width - 1)
        bottomleft = xy(transform, height - 1, 0)
        bottomright = xy(transform, height - 1, width - 1)

        id = file.split('.')[0]
        infofile = os.path.join(f'{sdata_dir}', id, f"{id}.json")
        with open(infofile, 'r') as f:
            data = json.load(f)
        print(f"Data: {data}")

        print("Top Left:", topleft)
        print("Top Right:", topright)
        print("Bottom Left:", bottomleft)
        print("Bottom Right:", bottomright)

        bbox = [
            topleft[0],  # minx
            bottomleft[1],  # miny
            topright[0],  # maxx
            topright[1]  # maxy
        ]

        center = [
            (topleft[0] + topright[0]) / 2,
            (topleft[1] + bottomleft[1]) / 2
        ]
        print(f"Center: {center[1]}, {center[0]}")

        wms_url = "https://kaart.maaamet.ee/wms/alus-geo?"

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": "of10000",  # või nt "pohi_vv" või "ortofoto"
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": ",".join(map(str, bbox)),
            "WIDTH": 800,
            "HEIGHT": 600,
            "FORMAT": "image/png",
            "TIME": data['chosenDate'][0],
            #"TIME": "2019-01-01T00:00:00Z",  # Asenda sobiva kuupäevaga
        }

        # Fetchi pilt
        response = requests.get(wms_url, params=params)
        print(f"Status Code: {response.status_code}")
        #print(f"Response: {response.text}")
        image = Image.open(BytesIO(response.content))

        plt.figure(figsize=(10, 7))
        plt.imshow(image)
        plt.axis("off")
        plt.title("Maa-ameti WMS kaart")
        plt.show()

parser = argparse.ArgumentParser(description="Fetch WMS image for a given file.")
parser.add_argument("file", type=str, help="The file to fetch WMS image for.")
args = parser.parse_args()

if args.file:
    fetch_wms_image(args.file)
else:
    for file in files:
        fetch_wms_image(file)
