import os
import gzip
import json
import csv
import numpy as np
import pandas as pd

from utils import printProgressBar

class ABO:
    def __init__(self, path_to_abo_dataset_folder):
        self.path_to_abo_dataset_folder = path_to_abo_dataset_folder
        self.gathered_data = {"main_image_id": [], "product_type": [], "color": [], "image_path": []}

    def read_metadata(self):
        i=0
        printProgressBar(i, 16, length=50)
        for json_file in os.listdir(self.path_to_abo_dataset_folder+"/listings/metadata"):
            f1 = os.path.join(self.path_to_abo_dataset_folder+"/listings/metadata",json_file)
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
            i+=1
            printProgressBar(i, 16, length=50)

    def map_metadata_to_images(self):
        images_metadata_df = pd.read_csv(self.path_to_abo_dataset_folder+"/images/metadata/images.csv.gz", compression='gzip')
        images_number=len(self.gathered_data["main_image_id"])
        printProgressBar(0,images_number, length=50)
        for i in range(images_number):
            self.gathered_data["image_path"].append(images_metadata_df[images_metadata_df["image_id"]==self.gathered_data["main_image_id"][i]]["path"].item())
            printProgressBar(i+1, images_number, length=50)

    def write_to_csv(self):
        with open("gathered_abo_data.csv", "w") as f:
            writer = csv.writer(f)
            key_list = list(self.gathered_data.keys())
            column_number = len(key_list)
            writer.writerow(self.gathered_data.keys())
            for feature in range(column_number):
                writer.writerow([self.gathered_data[key][feature] for key in key_list])

    def build_exploitable_dataset_from_raw_data(self):
        self.read_metadata()
        self.map_metadata_to_images()
        self.write_to_csv()

ABO("abo_dataset").build_exploitable_dataset_from_raw_data()
