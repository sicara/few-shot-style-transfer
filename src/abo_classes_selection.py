from typing import Union
from pathlib import Path
import pandas as pd


class ClassesSelection:
    def __init__(self, path_to_abo_dataset_folder: Union[Path, str]):
        self.path_to_abo_dataset_folder = Path(path_to_abo_dataset_folder)
        self.abo_classes = pd.DataFrame()
        self.imagenet_classes = pd.DataFrame()

    def get_consistent_abo_classes(self):
        pass

    def get_imagenet_classes(self):
        pass

    def show_matches(self):
        pass

    def write_selected_classes_to_json(self):
        pass

    def selected_classes_to_json(self):
        self.get_consistent_abo_classes()
        self.get_imagenet_classes()
        self.show_matches()
        self.write_selected_classes_to_json()
