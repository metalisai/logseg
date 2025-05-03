from sentinelhub import SHConfig, BBox, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
from shapely.geometry import Polygon
from overlay import add_overlay
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import math
from pyproj import Transformer
from find_overlaps import calculate_iou
import glob
import random

sdata_dir = 'sentinel_data_extra'

data = None
with open('output_data.json') as f:
    data = json.load(f)

config = SHConfig()
config.sh_client_id = 'c28f7723-a9cf-470c-8858-fa5415cf084e'
config.sh_client_secret = 'efxg6FjuDl0lbneD3q5locFC2NOYWyJu'

#poly = [[25.8923876347013, 58.420158472475705], [25.891374980754108, 58.42088420378719], [25.8915547054254, 58.42194707412148], [25.891628983128417, 58.4220675906556], [25.89229159198104, 58.42227200294367], [25.89290644203188, 58.420296747163555], [25.8923876347013, 58.420158472475705]]
#idx = 0
idxs = []
for idxx, item in enumerate(data):
    if item['aasta'] >= 2020 and item['tookood'] == 'LR':
        #idx = idxx
        idxs.append(idxx)
random.shuffle(idxs)

def findPolygonsInWindow(bounds, year):
    polys = []
    indices = []
    for idx in idxs:
        if data[idx]['aasta'] != year:
            continue
        poly = data[idx]['coords']
        overlap = False
        for p in poly:
            if p[0] >= bounds[0] and p[0] <= bounds[2] and p[1] >= bounds[1] and p[1] <= bounds[3]:
                overlap = True
                break
        if overlap:
            polys.append(poly)
            indices.append(idx)
    return polys, indices

if False:
    idxs = []
    desired = [2387520, 1692686, 1945100, 2264850, 2217994, 2264672, 1254678, 2390050, 11659162, 11638782, 11650096, 1529492, 2402272, 11741791, 11664105, 11725435, 1871868, 1373850, 2217960, 11613042, 1276046, 1945136, 2007996, 11620041, 1103312, 11664054, 11590354, 11593404, 11802006, 11745612, 1660038, 2069180, 11619936, 2288172, 1946178, 2387454, 2264220, 2264840, 1276030, 11811018, 11801753, 1370430, 11596092, 11632331, 11659818, 11641397, 1683818, 2001574, 2107270, 11741083, 2134214, 2228176, 11754715, 1630130, 11696141, 1250708, 2138130, 11801738, 11642229, 1394234, 11632313, 2387498, 11738229, 1878524, 1160986, 11601218, 2226258, 1507062, 2025084, 1945196, 2073818, 2007976, 1687022, 1686666, 2171362, 2404016, 1221830, 11697309, 1401710, 11725430, 2374856, 11664611, 1928038, 2157270, 2128628, 1708152, 11620369, 2121772, 2217884, 11696166, 1374326]
    for did in desired:
        for idx, item in enumerate(data):
            if item['id'] == did and item['tookood'] == 'LR' and item['aasta'] >= 2020:
                print(f"ADDING {did} as idx {idx}")
                idxs.append(idx)
                break

print(f"idxs: {idxs}")

def get_existing_bounds():
    imgs = glob.glob("output/*/*.tiff")
    ids = [os.path.basename(img).split(".")[0] for img in imgs]
    jsonfiles = [os.path.join("sentinel_data", id, f"{id}.json") for id in ids]

    ids2 = glob.glob(f"{sdata_dir}/*/*.json")
    ids2 = [os.path.basename(jsonfile).split(".")[0] for jsonfile in ids2]
    jsonfiles2 = [os.path.join(sdata_dir, id, f"{id}.json") for id in ids2]
    jsonfiles.extend(jsonfiles2)

    jsonfiles = list(set(jsonfiles))
    bboxes = []
    for jsonfile in jsonfiles:
        with open(jsonfile, 'r') as f:
            data = json.load(f)
            bbox = data['bbox']
            bboxes.append(bbox)
    return bboxes

existing_bounds = get_existing_bounds()

for idx in idxs:
    poly = data[idx]['coords']
    year = data[idx]['aasta']
    unitid = data[idx]['id']
    startDate1 = f"{year}-01-08"
    endDate1 = f"{year}-02-12"
    startDate2 = f"{year}-05-01"
    endDate2 = f"{year}-07-01"
    startDate3 = f"{year}-09-15"
    endDate3 = f"{year}-10-27"
    startDate4 = f"{year}-11-15"
    endDate4 = f"{year}-12-28"

    resolution = 10 # B2, B3, B4 have 10m resolution

    polygon = Polygon(poly)
    print(polygon.bounds)

    bounds = polygon.bounds
    center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    #side_meters = 256 * resolution
    side_meters = 128 * resolution
    # transformer to projected coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
    centerP = transformer.transform(center[0], center[1])
    bounds = (
        centerP[0] - side_meters / 2,
        centerP[1] - side_meters / 2,
        centerP[0] + side_meters / 2,
        centerP[1] + side_meters / 2
    )
    # transform back to lat/lon
    transformer = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
    mins = transformer.transform(bounds[0], bounds[1])
    maxs = transformer.transform(bounds[2], bounds[3])
    bounds = (mins[0], mins[1], maxs[0], maxs[1])

    ious = [calculate_iou(bounds, b) for b in existing_bounds]
    ious.sort(reverse=True)
    if len(ious) > 0 and ious[0] > 0.5:
        print(f"Skipping {unitid} due to overlap with existing data")
        continue

    overlayPolys, overlayIndices = findPolygonsInWindow(bounds, year)

    os.listdir(f'{sdata_dir}')
    if os.path.exists(f'{sdata_dir}/{data[idx]["id"]}'):
        print(f"Already processed {data[idx]['id']}")
        continue

    bbox = BBox(bbox=bounds, crs='EPSG:4326')
    bsize = bbox_to_dimensions(bbox, resolution=resolution)
    print(f"size: {bsize}")

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    evalscript_all_bands = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: [
                "B01", "B02", "B03", "B04", "B05", "B06", "B07",
                "B08", "B8A", "B09", B10, "B11", "B12"
            ],
            units: "DN"
        }],
        output: {
            bands: 12,
            sampleType: "UINT16"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B01, sample.B02, sample.B03, sample.B04,
        sample.B05, sample.B06, sample.B07, sample.B08,
        sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12
    ];
}
"""

    dates = [
        (startDate1, endDate1),
        (startDate2, endDate2),
        (startDate3, endDate3),
        (startDate4, endDate4)
    ]
    
    print(f"UNITID: {unitid}")

    outfolder = f'{sdata_dir}/{unitid}'
    # delete folder if exists
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)

    for date in dates:
        request = SentinelHubRequest(
            data_folder=os.path.join(outfolder, date[0]),
            evalscript=evalscript_true_color,
            #evalscript=evalscript_all_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=date,
                    maxcc=0.1,
                )
            ],
            bbox=bbox,
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            size=bsize,
            config=config
        )

        #response = request.get_data(save_data=True)
        request.save_data()

    dirs = os.listdir(outfolder)
    imgs = []
    for dirname in dirs:
        filepath = os.path.join(outfolder, dirname)
        indir = os.listdir(filepath)
        filepath = os.path.join(filepath, indir[0])
        filepath = os.path.join(filepath, 'response.tiff')
        #overlayPolys = [poly]
        img = add_overlay(overlayPolys, filepath)
        print(f"img shape: {img.shape}")
        imgs.append((img, dirname))

    imgs.sort(key=lambda x: x[1])
    comb = np.vstack([np.hstack([img[0] for img in imgs[:2]]), np.hstack([img[0] for img in imgs[2:4]])])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.imshow(comb)
    plt.show()

    choice = input("1,2,3,4,d:")
    delete = False
    if choice == 'd':
        delete = True
    else:
        choice = int(choice) - 1
        existing_bounds.append(bounds)
    chosenDate = dates[choice] if choice != 'd' else None

    infodict = {
        'unitid': unitid,
        'year': year,
        'coords': poly,
        'allPolys': overlayPolys,
        'imgs': [img[1] for img in imgs],
        'bbox': bounds,
        'center': center,
        'chosenDate': chosenDate,
        'delete': delete,
    }
    print(f"infodict: {infodict}")
    jsonfile = os.path.join(outfolder, f"{unitid}.json")
    with open(jsonfile, 'w') as f:
        json.dump(infodict, f)

    # save image
    outimg = os.path.join("examples", f"{unitid}.png")
    plt.imsave(outimg, comb)

    plt.close()
