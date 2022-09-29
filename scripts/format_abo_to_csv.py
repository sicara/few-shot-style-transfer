from src.abo_data_formatting import ABOFormatting
from src.config import ROOT_FOLDER

ABOFormatting(ROOT_FOLDER / "abo_dataset").build_metadata_csv_from_raw_data()
