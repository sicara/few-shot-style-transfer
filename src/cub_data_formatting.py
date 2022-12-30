import pandas as pd
from pathlib import Path
from typing import Union

from src.config import ROOT_FOLDER


class CUBFormatting:
    def __init__(self, path_to_cub_dataset_folder: Union[Path, str]):
        self.path_to_cub_dataset_folder = Path(path_to_cub_dataset_folder)
        self.gathered_data_df = pd.DataFrame(
            columns=[
                "image_id",
                "path",
                "class_name",
                "class_id",
                "color_attribute_id",
                "color",
            ]
        )

    def build_metadata_csv_from_raw_data(self):
        images_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("images.txt"),
            sep=" ",
            header=None,
            names=["image_id", "path"],
        )
        images_df["class_name"] = images_df.apply(
            lambda row: str(row["path"]).split("/")[0], axis=1
        )
        classes_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("classes.txt"),
            sep=" ",
            header=None,
            names=["class_id", "class_name"],
        )
        class_atributes_df = pd.read_csv(
            self.path_to_cub_dataset_folder
            / Path("attributes/class_attribute_labels_continuous.txt"),
            sep=" ",
            header=None,
            names=[str(i) for i in range(1, 313)],
        )
        class_atributes_df["class_id"] = class_atributes_df.index
        class_atributes_df["class_id"] = class_atributes_df.apply(
            lambda row: int(row.class_id + 1), axis=1
        )
        class_atributes_df["color_attribute_id"] = class_atributes_df[
            [str(i) for i in range(249, 264)]
        ].idxmax(axis=1)
        attribute_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("classes.txt"),
            sep=" ",
            header=None,
            names=["attribute_id", "attribute_name"],
        )
        attribute_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("attributes/attributes.txt"),
            sep=" ",
            header=None,
            names=["attribute_id", "attribute_name"],
        )
        attribute_df["color"] = attribute_df.apply(
            lambda row: str(row["attribute_name"]).split("::")[-1], axis=1
        )
        attribute_df["attribute_id"] = attribute_df.apply(
            lambda row: str(row.attribute_id), axis=1
        )

        self.gathered_data_df = (
            images_df.merge(classes_df, how="left", on="class_name")
            .merge(
                class_atributes_df[["color_attribute_id", "class_id"]],
                how="left",
                on="class_id",
            )
            .merge(
                attribute_df[["attribute_id", "color"]],
                how="left",
                left_on="color_attribute_id",
                right_on="attribute_id",
            )
            .drop("attribute_id", axis=1)
        )
        # TODO: add color label for each image (not only for the class label)
        ## from image_attribute_labels.txt
        self.gathered_data_df.to_csv(
            ROOT_FOLDER / "data/gathered_cub_data.csv", index=False
        )
