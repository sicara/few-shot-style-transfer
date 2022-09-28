import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class ABOFormatting:
    def __init__(self, path_to_abo_dataset_folder):
        self.path_to_abo_dataset_folder = path_to_abo_dataset_folder
        self.metadata = pd.DataFrame()
        self.gathered_data = pd.DataFrame()

    def read_metadata(self):
        print("Metadata collection:")
        metadata_dict = {"main_image_id": [], "product_type": [], "color": []}
        json_files = list(Path(self.path_to_abo_dataset_folder + "/listings/metadata").iterdir())
        for json_file in tqdm(json_files):
            with gzip.open(f"{json_file}", "r") as f:
                data = [json.loads(line) for line in f]
            for product in data:
                if "main_image_id" in product:
                    metadata_dict["main_image_id"].append(product["main_image_id"])
                    metadata_dict["product_type"].append(product["product_type"][0]["value"])
                    metadata_dict["color"].append(
                        product["color"][0]["standardized_values"][0]
                        if ("color" in product and "standardized_values" in product["color"][0])
                        else np.nan
                    )
        self.metadata = pd.DataFrame.from_dict(metadata_dict, orient="columns")
        self.metadata.set_index("main_image_id", inplace=True)

    def map_metadata_to_images(self):
        images_metadata_df = pd.read_csv(
            self.path_to_abo_dataset_folder + "/images/metadata/images.csv.gz", compression="gzip"
        )
        self.gathered_data = pd.merge(
            self.metadata, images_metadata_df, how="left", left_on="main_image_id", right_on="image_id"
        )[["product_type", "color", "image_id", "path"]]
        self.gathered_data.set_index("image_id", inplace=True)

    def build_metadata_csv_from_raw_data(self):
        self.read_metadata()
        self.map_metadata_to_images()
        self.gathered_data.to_csv("gathered_abo_data_a.csv")
