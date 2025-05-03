from sentinelhub import SHConfig, BBox, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
import glob
import os
import json
import shutil

config = SHConfig()
config.sh_client_id = 'c28f7723-a9cf-470c-8858-fa5415cf084e'
config.sh_client_secret = 'efxg6FjuDl0lbneD3q5locFC2NOYWyJu'

image_dir = '../train/images'
refetch_dir = '../train/images_brighttest'

files = glob.glob(f"{image_dir}/*/*.tiff")

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
        return [sample.B02, sample.B03, sample.B04];
    }
"""

evalscript_false_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B03", "B04", "B08"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B08, sample.B04, sample.B03];
    }
"""

evalscript_all_bands = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: [
                "B01", "B02", "B03", "B04", "B05", "B06", "B07",
                "B08", "B8A", "B09", "B10", "B11", "B12"
            ],
        }],
        output: {
            bands: 13,
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B01, sample.B02, sample.B03, sample.B04,
        sample.B05, sample.B06, sample.B07, sample.B08,
        sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12
    ];
}"""


for file in files:
    parts = file.strip(os.sep).split(os.sep)
    new_path = os.path.join(*parts[1:])
    new_path = os.path.join(refetch_dir, new_path)

    eid = parts[-1].split(".")[0]
    print(f"{eid}")

    metadata_file = os.path.join("sentinel_data_all", eid, f"{eid}.json")
    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {metadata_file} not found. Skipping file {file}.")
        continue

    bounds = data['bbox']
    start_date = data['chosenDate'][0]
    end_date = data['chosenDate'][1]

    print(f"Processing {new_path}")

    bbox = BBox(bbox=bounds, crs='EPSG:4326')

    outfolder = f'refetch_data/{eid}'
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)

    request = SentinelHubRequest(
        data_folder=outfolder,
        evalscript=evalscript_true_color,
        #evalscript=evalscript_false_color,
        #evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
                maxcc=0.1,
            )
        ],
        bbox=bbox,
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        size=(128, 128),
        config=config
    )

    request.save_data()

    # find the output folder
    outpath = os.listdir(outfolder)[0]
    outpath = os.path.join(outfolder, outpath)
    outpath = os.path.join(outpath, 'response.tiff')
    # create the new folder if it doesn't exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    result = os.system(f"cp {outpath} {new_path}")
    if result == 0:
        print(f"Copied {outpath} to {new_path}")
    else:
        print(f"Failed to copy {outpath} to {new_path}")
