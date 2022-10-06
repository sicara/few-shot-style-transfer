from typing import Union
from pathlib import Path
import pandas as pd
import re
import json

from src.config import ROOT_FOLDER


class ClassesSelection:
    def __init__(
        self,
        path_to_abo_dataset_csv: Union[Path, str],
        path_to_imagenet_csv: Union[Path, str],
    ):
        self.path_to_abo_dataset_csv = Path(path_to_abo_dataset_csv)
        self.path_to_imagenet_csv = Path(path_to_imagenet_csv)
        self.abo_classes = []
        self.imagenet_classes = []
        self.matched_classes = []

    def get_consistent_abo_classes(self):
        abo_class_df = (
            pd.read_csv(self.path_to_abo_dataset_csv)
            .groupby("product_type")["product_type"]
            .count()
            .reset_index(name="count")
        )
        self.abo_classes = list(
            abo_class_df[abo_class_df["count"] >= 17]["product_type"]
        )

    def get_imagenet_classes(self):
        imagenet_class_df = pd.read_csv(self.path_to_imagenet_csv, sep="|")
        self.imagenet_classes = list(
            imagenet_class_df.assign(
                **{"Class Name": imagenet_class_df["Class Name"].str.split(",")}
            )
            .explode("Class Name")
            .apply(lambda row: row["Class Name"].lower(), axis=1)
        )

    def ask_for_user_input(self, abo_class, imagenet_class, information_sentence=""):
        print(abo_class, "--", imagenet_class)
        user_answer = input(information_sentence + "Is it a match (y/n)? ")
        if user_answer.lower() == "y":
            self.matched_classes.append(abo_class)
            return True
        elif user_answer.lower() != "n":
            self.ask_for_user_input(
                abo_class,
                imagenet_class,
                information_sentence="Only 'y' and 'n' answers are allowed.",
            )
        else:
            return False

    def get_matches_between_imagenet_and_abo(self):
        """
        Classes are matched based on:
        - equality of strings (ex: 'sleep bag' is equal to 'sleep bag' not to 'bag') or
        - matching of words not subset of words (ex: 'bag' is matched with 'sleep bag' but not with 'bagger')
        """
        for abo_class in self.abo_classes:
            for imagenet_class in self.imagenet_classes:
                if str(abo_class).lower().replace("_", " ") == str(imagenet_class):
                    self.matched_classes.append(abo_class)
                    break
                elif re.search(
                    r"(?:^|\W)"
                    + (str(abo_class).lower().replace("_", " "))
                    + r"(?:$|\W)",
                    str(imagenet_class),
                ):
                    stop_bool = self.ask_for_user_input(abo_class, imagenet_class)
                    if stop_bool:
                        break

    def write_selected_classes_to_json(self):
        selected_classes = list(set(self.abo_classes) - set(self.matched_classes))
        with open(
            ROOT_FOLDER / "src/selected_and_matched_abo_classes_new.json", "w"
        ) as f:
            json.dump(
                {
                    "matched": self.matched_classes,
                    "selected": selected_classes,
                    "manually_eliminated": [],
                },
                f,
            )

    def selected_classes_to_json(self):
        self.get_consistent_abo_classes()
        self.get_imagenet_classes()
        self.get_matches_between_imagenet_and_abo()
        self.write_selected_classes_to_json()
