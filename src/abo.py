from typing import Callable, Tuple, List, Union
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from PIL import Image

from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import default_transform
from src.config import ROOT_FOLDER


class ABO(FewShotDataset):
    def __init__(
        self,
        root: Union[Path, str],
        specs_file: Union[Path, str] = ROOT_FOLDER / "src" / "datasets" / "gathered_abo_data.csv",
        image_size: int = 84,
        transform: Callable = None,
        training: bool = False,
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
        """
        self.root = ROOT_FOLDER / root
        self.data = self.load_specs(specs_file)
        self.class_names = list(self.data["product_type"].unique())
        self.transform = transform if transform else default_transform(image_size, training=training)

    @staticmethod
    def load_specs(specs_file: Union[Path, str]) -> DataFrame:
        data = pd.read_csv(specs_file)
        class_names = list(data["product_type"].unique())

        label_mapping = {name: class_names.index(name) for name in class_names}

        return data.assign(label=lambda df: df["product_type"].map(label_mapping))

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        img = self.transform(Image.open(self.root / self.data.path[item]).convert("RGB"))
        label = self.data.label[item]

        return img, label

    def __len__(self) -> int:
        return len(self.data)

    def get_labels(self) -> List[int]:
        return list(self.data.label)
