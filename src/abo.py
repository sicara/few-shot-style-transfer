from optparse import Option
from typing import Callable, Tuple, List, Union, Optional
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from PIL import Image
import json
from loguru import logger

from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import default_transform
from src.config import ROOT_FOLDER


class ABO(FewShotDataset):
    def __init__(
        self,
        root: Union[Path, str],
        specs_file: Union[Path, str] = ROOT_FOLDER / "data" / "gathered_abo_data.csv",
        image_size: int = 84,
        transform: Optional[Callable] = None,
        training: bool = False,
        classes_json: Optional[Path] = None,
        colors_json: Optional[Path] = None,
        min_number_item_per_class: int = 17,
    ):
        """
        Args:
            root: directory where all the images are
            specs_file: path to the CSV file
            image_size: images returned by the dataset will be square images of the given size
            transform: torchvision transforms to be applied to images. If none is provided,
                we use some standard transformations including ImageNet normalization.
                These default transformations depend on the "training" argument.
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip. Only used if transforms = None.
            classes_json: path to the json file containing the selected classes. If no path is given, all the classes are used.
            colors_json: path to the json file containing the selected colors. If no path is given, all the colors are used.
        """
        super().__init__()
        self.root = ROOT_FOLDER / root
        self.data, self.class_names = self.load_specs_and_classes(
            specs_file, classes_json, colors_json, min_number_item_per_class
        )
        self.transform = (
            transform if transform else default_transform(image_size, training=training)
        )
        self.min_number_of_item_per_class = min_number_item_per_class

    @staticmethod
    def load_specs_and_classes(
        specs_file: Union[Path, str],
        classes_json: Optional[Path],
        colors_json: Optional[Path],
        min_number_item_per_class: int,
    ) -> Tuple[DataFrame, List[str]]:
        data = pd.read_csv(specs_file).drop_duplicates(subset=["path"])
        if colors_json is not None:
            with open(ROOT_FOLDER / colors_json) as json_file:
                data = data[data.en_color.isin(json.load(json_file)["selected"])]
        if classes_json is not None:
            with open(ROOT_FOLDER / classes_json) as json_file:
                data = data[data.product_type.isin(json.load(json_file)["selected"])]
        data_product_type_count = pd.DataFrame(
            {
                "product_type": data["product_type"].value_counts().index,
                "count": data["product_type"].value_counts().values,
            }
        )
        class_names = list(
            data_product_type_count[
                data_product_type_count["count"] >= min_number_item_per_class
            ]["product_type"]
        )
        removed_classes = list(
            data_product_type_count[
                data_product_type_count["count"] < min_number_item_per_class
            ]["product_type"]
        )
        if len(removed_classes) > 0:
            logger.info(
                f"Removed classes {removed_classes} because they had less than {str(min_number_item_per_class)} elements."
            )
        data = data[data.product_type.isin(class_names)].reset_index()
        label_mapping = {name: class_names.index(name) for name in class_names}

        return (
            data.assign(label=lambda df: df["product_type"].map(label_mapping)),
            class_names,
        )

    def __getitem__(self, item: int) -> Tuple[Tensor, int, str, str]:
        img = self.transform(
            Image.open(self.root / self.data.path[item]).convert("RGB")
        )
        label = self.data.label[item]
        color = self.data.en_color[item]
        img_path = self.data.path[item]

        return img, label, color, img_path

    def get_item_label(self, item: int) -> int:
        return self.data.label[item]

    def get_item_color(self, item: int) -> str:
        return self.data.en_color[item]

    def get_item_img_path(self, item: int) -> str:
        return self.data.path[item]

    def __len__(self) -> int:
        return len(self.data)

    def get_labels(self) -> List[int]:
        return list(self.data.label)

    def get_colors(self) -> List[str]:
        return list(self.data.en_color)

    def get_img_path(self) -> List[str]:
        return list(self.data.path)
