import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Union

from src.config import ROOT_FOLDER


class ABOFormatting:
    def __init__(self, path_to_abo_dataset_folder: Union[Path, str]):
        self.path_to_abo_dataset_folder = path_to_abo_dataset_folder
        self.metadata_df = pd.DataFrame()
        self.gathered_data_df = pd.DataFrame()

    def read_metadata(self):
        metadata_dict = {"main_image_id": [], "product_type": [], "color": []}
        json_files = list((self.path_to_abo_dataset_folder / "listings" / "metadata").iterdir())
        for json_file in tqdm(json_files, desc="Metadata collection"):
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
        self.metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="columns")
        self.metadata_df.set_index("main_image_id", inplace=True)

    def map_metadata_to_images(self):
        images_metadata_df = pd.read_csv(
            self.path_to_abo_dataset_folder / "images/metadata/images.csv.gz", compression="gzip"
        )
        self.gathered_data_df = pd.merge(
            self.metadata_df, images_metadata_df, how="left", left_on="main_image_id", right_on="image_id"
        )[["product_type", "color", "image_id", "path"]]
        self.gathered_data_df.set_index("image_id", inplace=True)

    def build_metadata_csv_from_raw_data(self):
        self.read_metadata()
        self.map_metadata_to_images()
        self.gathered_data_df.to_csv(ROOT_FOLDER / "src/datasets/gathered_abo_data.csv")
