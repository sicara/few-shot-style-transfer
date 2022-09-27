import os
import gzip
import json
import numpy as np

class ABO:
    def __init__(self, path_to_abo_dataset_folder):
        self.path_to_abo_dataset_folder = path_to_abo_dataset_folder
        self.gathered_data = {"main_image_id": [], "product_type": [], "color": [], "image_path": []}

    def read_metadata(self):
        for json_file in os.listdir(self.path_to_abo_dataset_folder+"/listings/metadata"):
            f = os.path.join(self.path_to_abo_dataset_folder+"/listings/metadata",json_file)
            print(json_file)
            with gzip.open(self.path_to_abo_dataset_folder+"/listings/metadata/"+json_file, 'r') as f:
                data = [json.loads(line) for line in f]
            for product in data:
                if "main_image_id" in product:
                    self.gathered_data["main_image_id"].append(product["main_image_id"])
                    self.gathered_data["product_type"].append(product["product_type"][0]["value"])
                    if "color" in product:
                        if "standardized_values" in product['color'][0]:
                            self.gathered_data["color"].append(product['color'][0]['standardized_values'][0])
                        else:
                            self.gathered_data["color"].append(np.nan)
                    else:
                        self.gathered_data["color"].append(np.nan)

    def map_metadata_to_images(self):
        pass

    def build_exploitable_dataset_from_raw_data(self):
        self.read_metadata()
        self.map_metadata_to_images()
        print(self.gathered_data["main_image_id"][0:5], self.gathered_data["product_type"][0:5], self.gathered_data["color"][0:5], self.gathered_data["image_path"][0:5])

ABO("abo_dataset").build_exploitable_dataset_from_raw_data()
