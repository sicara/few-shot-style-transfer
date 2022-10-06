from src.abo_classes_selection import ClassesSelection
from src.config import ROOT_FOLDER

ClassesSelection(
    ROOT_FOLDER / "data" / "gathered_abo_data.csv",
    ROOT_FOLDER / "data" / "imagenet_classes.csv",
).selected_classes_to_json()
