from src.abo_classes_selection import ClassesSelection
from src.config import ROOT_FOLDER

ClassesSelection(
    ROOT_FOLDER / "src" / "datasets" / "gathered_abo_data.csv",
    ROOT_FOLDER / "src" / "datasets" / "imagenet_classes.csv",
).selected_classes_to_json()
