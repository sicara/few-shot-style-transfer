from typing import Callable, Tuple
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from PIL import Image

from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import default_transform

class ABO(FewShotDataset):
    def __init__(
        self,
        root: str,
        specs_file: str = "gathered_abo_data.csv",
        image_size: int = 84,
        transform: Callable = None,
        training: bool = False,
    ):
        self.root = "abo_dataset/images/small"
        self.data = self.load_specs(specs_file)
        self.class_names = list(self.data["product_type"].unique())
        self.transform = (transform if transform else default_transform(image_size, training=training))

    def load_specs(self, specs_file: str) -> DataFrame:
        data = pd.read_csv(specs_file)

        label_mapping = {name: self.class_names.index(name) for name in self.class_names}

        return data.assign(label=lambda df: df["product_type"].map(label_mapping))

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        img = self.transform(
            Image.open(self.root+"/"+self.data.image_path[item]).convert("RGB")
        )
        label = self.data.label[item]

        return img, label
