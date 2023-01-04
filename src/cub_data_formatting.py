import pandas as pd
from pathlib import Path
from typing import Union

from src.config import ROOT_FOLDER

PRIMARY_COLOR_ATTRIBUTE_ID = (249, 263)
ANNOTATION_CERTAINTY_THRESHOLD = 3
ANNOTATION_PRESENCE = 1


class CUBFormatting:
    def __init__(self, path_to_cub_dataset_folder: Union[Path, str]):
        self.path_to_cub_dataset_folder = Path(path_to_cub_dataset_folder)
        self.image_df = pd.DataFrame()
        self.classes_df = pd.DataFrame()
        self.class_attributes_df = pd.DataFrame()
        self.attributes_df = pd.DataFrame()
        self.image_annotation_df = pd.DataFrame()
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

    def read_and_format_images_data(self):
        self.images_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("images.txt"),
            sep=" ",
            header=None,
            names=["image_id", "path"],
        )
        self.images_df["class_name"] = self.images_df.apply(
            lambda row: str(row["path"]).split("/")[0], axis=1
        )

    def read_and_format_classes_data(self):
        self.classes_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("classes.txt"),
            sep=" ",
            header=None,
            names=["class_id", "class_name"],
        )

    def read_and_format_classes_attributes_data(self):
        self.class_atributes_df = pd.read_csv(
            self.path_to_cub_dataset_folder
            / "attributes/class_attribute_labels_continuous.txt",
            sep=" ",
            header=None,
            names=[str(i) for i in range(1, 313)],
        )
        self.class_atributes_df["class_id"] = self.class_atributes_df.index
        self.class_atributes_df["class_id"] = self.class_atributes_df.apply(
            lambda row: int(row.class_id + 1), axis=1
        )
        self.class_atributes_df["color_attribute_id"] = self.class_atributes_df[
            [
                str(i)
                for i in range(
                    PRIMARY_COLOR_ATTRIBUTE_ID[0], PRIMARY_COLOR_ATTRIBUTE_ID[1] + 1
                )
            ]
        ].idxmax(axis=1)

    def read_and_format_attributes_data(self):
        self.attribute_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("classes.txt"),
            sep=" ",
            header=None,
            names=["attribute_id", "attribute_name"],
        )
        self.attribute_df = pd.read_csv(
            self.path_to_cub_dataset_folder / Path("attributes/attributes.txt"),
            sep=" ",
            header=None,
            names=["attribute_id", "attribute_name"],
        )
        self.attribute_df["color"] = self.attribute_df.apply(
            lambda row: str(row["attribute_name"]).split("::")[-1], axis=1
        )
        self.attribute_df["attribute_id"] = self.attribute_df.apply(
            lambda row: str(row.attribute_id), axis=1
        )

    def read_and_format_images_annotation_data(self):
        self.image_annotation_df = pd.read_csv(
            self.path_to_cub_dataset_folder
            / Path("attributes/image_attribute_labels.txt"),
            sep=" ",
            header=None,
            names=[
                "image_id",
                "annotated_attribute_id",
                "is_present",
                "certainty",
                "time",
            ],
            on_bad_lines="warn",
        )
        self.image_annotation_df = (
            self.image_annotation_df[
                (
                    self.image_annotation_df["annotated_attribute_id"]
                    >= PRIMARY_COLOR_ATTRIBUTE_ID[0]
                )
                & (
                    self.image_annotation_df["annotated_attribute_id"]
                    <= PRIMARY_COLOR_ATTRIBUTE_ID[1]
                )
                & (
                    self.image_annotation_df["certainty"]
                    >= ANNOTATION_CERTAINTY_THRESHOLD
                )
                & (self.image_annotation_df["is_present"] == ANNOTATION_PRESENCE)
            ]
            .groupby("image_id")
            .first()
            .reset_index()
        )

    def build_metadata_csv_from_raw_data(self):
        self.read_and_format_images_data()
        self.read_and_format_classes_data()
        self.read_and_format_classes_attributes_data()
        self.read_and_format_attributes_data()
        self.read_and_format_images_annotation_data()
        self.gathered_data_df = (
            self.images_df.merge(self.classes_df, how="left", on="class_name")
            .merge(
                self.image_annotation_df[["image_id", "annotated_attribute_id"]],
                how="left",
                on="image_id",
            )
            .merge(
                self.class_atributes_df[["color_attribute_id", "class_id"]],
                how="left",
                on="class_id",
            )
            .merge(
                self.attribute_df[["attribute_id", "color"]],
                how="left",
                left_on="color_attribute_id",
                right_on="attribute_id",
            )
            .drop("attribute_id", axis=1)
        )
        self.gathered_data_df.annotated_attribute_id.fillna(
            self.gathered_data_df.color_attribute_id, inplace=True
        )
        self.gathered_data_df = self.gathered_data_df.astype(
            {"annotated_attribute_id": "int"}
        )
        # TODO: add color label for each image (not only for the class label)
        ## from image_attribute_labels.txt
        self.gathered_data_df.to_csv(
            ROOT_FOLDER / "data/gathered_cub_data.csv", index=False
        )
