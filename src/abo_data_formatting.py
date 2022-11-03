import gzip
import json
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Union
from googletrans import Translator

from src.config import ROOT_FOLDER


class ABOFormatting:
    def __init__(self, path_to_abo_dataset_folder: Union[Path, str]):
        self.path_to_abo_dataset_folder = Path(path_to_abo_dataset_folder)
        self.metadata_df = pd.DataFrame()
        self.gathered_data_df = pd.DataFrame()
        self.translator = Translator()

    def read_metadata(self):
        metadata_dict = {
            "main_image_id": [],
            "product_type": [],
            "color": [],
            "language": [],
        }
        json_files = list(
            (self.path_to_abo_dataset_folder / "listings" / "metadata").iterdir()
        )
        for json_file in tqdm(json_files, desc="Metadata collection"):
            with gzip.open(f"{json_file}", "r") as f:
                data = [json.loads(line) for line in f]
            for product in data:
                if "main_image_id" in product:
                    metadata_dict["main_image_id"].append(product["main_image_id"])
                    metadata_dict["product_type"].append(
                        product["product_type"][0]["value"]
                    )
                    metadata_dict["color"].append(
                        re.sub(
                            r"[^a-zA-Z]",
                            "",
                            product["color"][0]["standardized_values"][0],
                        )
                        if (
                            "color" in product
                            and "standardized_values" in product["color"][0]
                        )
                        else np.nan
                    )
                    metadata_dict["language"].append(
                        product["color"][0]["language_tag"][:2]
                        if ("color" in product)
                        else np.nan
                    )
        self.metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="columns")

    def translation_from_unknown_language(self, text):
        try:
            language_detected = self.translator.detect(text + " " + text).lang
            if language_detected != "en":
                return self.translation_to_en(text, language_detected)
            else:
                return re.sub(r"[^a-zA-Z]", "", text.lower())
        except:
            return np.nan

    def translation_to_en(self, text, src_language):
        if len(text) > 1:
            if src_language != "en":
                try:
                    return re.sub(
                        r"[^a-zA-Z]",
                        "",
                        self.translator.translate(
                            text, dest="en", src=src_language
                        ).text.lower(),
                    )
                except IndexError:
                    try:
                        return re.sub(
                            r"[^a-zA-Z]",
                            "",
                            self.translator.translate(text, dest="en").text.lower(),
                        )
                    except:
                        return np.nan
                except ValueError:
                    # unknown src language
                    return self.translation_from_unknown_language(text)
                except AttributeError:
                    # too many requests
                    try:
                        return re.sub(
                            r"[^a-zA-Z]",
                            "",
                            self.translator.translate(
                                text, dest="en", src=src_language
                            ).text.lower(),
                        )
                    except:
                        return self.translation_to_en(text, src_language)
            else:
                return self.translation_from_unknown_language(text)
        else:
            return np.nan

    def uniformize_color_names(self):
        tqdm.pandas(desc="Color homogenization")
        # color_grouped_df = self.metadata_df.groupby(["color", "language"])["color"].count().reset_index(name="count")
        # color_grouped_df["en_color"] = color_grouped_df.progress_apply(
        #    lambda row: self.translation_to_en(row.color, row.language), axis=1
        # )
        color_grouped_df = (
            self.metadata_df.groupby(["color", "language"])["color"]
            .count()
            .reset_index(name="count")
            .assign(
                en_color=lambda df: df.progress_apply(
                    lambda row: self.translation_to_en(row.color, row.language), axis=1
                )
            )
        )
        self.metadata_df = pd.merge(
            self.metadata_df, color_grouped_df, on=["color"], how="left"
        )
        self.metadata_df["en_color"] = self.metadata_df["en_color"].replace(
            {
                "gray": "grey",
                "multi": "multicolored",
                "multicolor": "multicolored",
                "multicolour": "multicolored",
                "multicoloured": "multicolored",
                "multicolourated": "multicolored",
                "goldroeseilver": "goldrosesilver",
                "golden": "gold",
                "navy": "blue",
                "nero": "black",
            }
        )

    def map_metadata_to_images(self):
        images_metadata_df = pd.read_csv(
            self.path_to_abo_dataset_folder / "images/metadata/images.csv.gz",
            compression="gzip",
        )
        self.gathered_data_df = pd.merge(
            self.metadata_df,
            images_metadata_df,
            how="left",
            left_on="main_image_id",
            right_on="image_id",
        )[["product_type", "en_color", "image_id", "path"]]
        self.gathered_data_df.set_index("image_id", inplace=True)

    def build_metadata_csv_from_raw_data(self):
        self.read_metadata()
        self.uniformize_color_names()
        self.map_metadata_to_images()
        self.gathered_data_df.to_csv(
            ROOT_FOLDER / "data/gathered_abo_data_color.csv", index=False
        )
