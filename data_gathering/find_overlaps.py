import os
import json
from itertools import combinations

data_folder = 'sentinel_data'

eraldised = os.listdir(data_folder)
bboxes = []
for folder in eraldised:
    folder_path = os.path.join(data_folder, folder)
    jsonfile = os.path.join(folder_path, f"{folder}.json")
    try:
        with open(jsonfile, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        #print(f"File {jsonfile} not found. Skipping folder {folder}.")
        continue

    #print(f"bounds: {data['bbox']}")
    bbox = data['bbox']
    bboxes.append((bbox, folder))

def calculate_iou(bbox1, bbox2):
    # Find coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Check if there is no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Areas of the bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Area of union
    union_area = bbox1_area + bbox2_area - intersection_area

    # IoU
    iou = intersection_area / union_area

    return iou

exceptions = [
    ("11620376", "11620369"),
    ("11650096", "1394234"),
    ("11650096", "11664611"),
    ("11650096", "2073818"),
    ("11801738", "11801753"),
]

pairs = list(combinations(bboxes, 2))
for pair in pairs:
    bbox1 = pair[0][0]
    bbox2 = pair[1][0]
    bbox1_folder = pair[0][1]
    bbox2_folder = pair[1][1]
    # check if they overlap
    if (bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0] and
            bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1]):
        # check if they exist in output folder
        output_folder = "output"
        bbox1_path = os.path.join(output_folder, f"{bbox1_folder}.tiff")
        bbox2_path = os.path.join(output_folder, f"{bbox2_folder}.tiff")
        isException = False
        for exception in exceptions:
            if bbox1_folder in exception and bbox2_folder in exception:
                isException = True
                break

        iou = calculate_iou(bbox1, bbox2)
        if os.path.exists(bbox1_path) and os.path.exists(bbox2_path) and not isException and iou > 0.5:
            #print(f"Overlap between {bbox1_folder} and {bbox2_folder}: {iou:.2f}")
            #print(f"{bbox1_folder}")
            print(f"{bbox2_folder}")
