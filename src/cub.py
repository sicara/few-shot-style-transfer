from pathlib import Path
import pandas as pd
from torch import Tensor
from PIL import Image
from typing import Callable, List, Optional, Tuple, Union
from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import default_transform
from src.config import ROOT_FOLDER


class CUB(FewShotDataset):
    def __init__(
        self,
        root: Union[Path, str],
        specs_file: Union[Path, str] = ROOT_FOLDER / "data" / "gathered_cub_data.csv",
        image_size: int = 84,
        transform: Optional[Callable] = None,
        training: bool = False,
    ):
        super().__init__()
        self.root = ROOT_FOLDER / root
        self.data = self.load_specs_and_classes(specs_file)
        self.transform = (
            transform if transform else default_transform(image_size, training=training)
        )

    @staticmethod
    def load_specs_and_classes(specs_file):
        data = pd.read_csv(specs_file)
        return data

    def __getitem__(self, item: int) -> Tuple[Tensor, int, str, str]:
        img = self.transform(
            Image.open(self.root / self.data.path[item]).convert("RGB")
        )
        label = self.data.class_id[item]
        color = self.data.annotated_attribute_id[item]
        img_path = self.data.path[item]

        return img, label, color, img_path

    def get_item_label(self, item: int) -> int:
        return self.data.class_id[item]

    def get_item_color(self, item: int) -> str:
        return self.data.annotated_attribute_id[item]

    def get_item_img_path(self, item: int) -> str:
        return self.data.path[item]

    def __len__(self) -> int:
        return len(self.data)

    def get_labels(self) -> List[int]:
        return list(self.data.class_id)

    def get_colors(self) -> List[str]:
        return list(self.data.annotated_attribute_id)

    def get_img_path(self) -> List[str]:
        return list(self.data.path)
