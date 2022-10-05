from typing import Union
from pathlib import Path
import pandas as pd
import re

from src.config import ROOT_FOLDER


class ClassesSelection:
    def __init__(self, path_to_abo_dataset_csv: Union[Path, str], path_to_imagenet_csv: Union[Path, str]):
        self.path_to_abo_dataset_csv = Path(path_to_abo_dataset_csv)
        self.path_to_imagenet_csv = Path(path_to_imagenet_csv)
        self.abo_classes = pd.DataFrame()
        self.imagenet_classes = pd.DataFrame()
        self.matched_classes = []

    def get_consistent_abo_classes(self):
        data = (
            pd.read_csv(self.path_to_abo_dataset_csv)
            .groupby("product_type")["product_type"]
            .count()
            .reset_index(name="count")
        )
        self.abo_classes = list(
            data[data["count"] >= 17].apply(lambda row: row.product_type.lower().replace("_", " "), axis=1)
        )

    def get_imagenet_classes(self):
        data = pd.read_csv(self.path_to_imagenet_csv, sep="|")
        self.imagenet_classes = list(
            data.assign(**{"Class Name": data["Class Name"].str.split(",")})
            .explode("Class Name")
            .apply(lambda row: row["Class Name"].lower(), axis=1)
        )

    def show_matches(self):
        pass

    def write_selected_classes_to_json(self):
        pass

    def selected_classes_to_json(self):
        self.get_consistent_abo_classes()
        self.get_imagenet_classes()
        self.show_matches()
        self.write_selected_classes_to_json()


ClassesSelection(
    Path(ROOT_FOLDER / "src" / "datasets" / "gathered_abo_data.csv"),
    Path(ROOT_FOLDER / "src" / "datasets" / "imagenet_classes.csv"),
).selected_classes_to_json()
