import os
import json
import geopandas as gpd
import json

resp_path = "responses"

gdf = gpd.read_file("eraldis.shp")
print(f"CRS: {gdf.crs}")
gdf = gdf.to_crs(epsg=4326)

output_data = []

files = os.listdir("responses")
for file in files:
    response_path = os.path.join(resp_path, file)
    print(f"Processing {response_path}")
    with open(response_path, "r") as f:
        try:
            response = json.load(f)
            if len(response['tood']) != 0:
                id = response['id']
                #print(f"Response is not empty: {response['tood']}")
                for too in response['tood']:
                    tookood = too['tooKood']
                    if "aasta" in too:
                        aasta = too['aasta']
                        print(f"ID: {id}, tooKood: {tookood}, aasta: {too['aasta']}")
                        georow = gdf[gdf['ID'] == id].iloc[0]
                        print(f"Georow: {georow}")
                        geometry = georow['geometry']
                        if geometry.geom_type == 'Polygon':
                            xx, yy = geometry.convex_hull.exterior.coords.xy
                            coords = list(zip(xx, yy))
                            print(f"{id} {tookood} {aasta} {coords}")
                            data = {
                                "file": file,
                                "id": id,
                                "tookood": tookood,
                                "aasta": aasta,
                                "coords": coords
                            }
                            output_data.append(data)
                        else:
                            print(f"Unknown geometry type: {geometry.geom_type}")
                    
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {response_path}")
            continue

# Save the output data to a JSON file
output_file = "output_data.json"
with open(output_file, "w") as f:
    json.dump(output_data, f)
