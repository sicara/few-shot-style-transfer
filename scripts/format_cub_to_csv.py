from src.cub_data_formatting import CUBFormatting
from src.config import ROOT_FOLDER

CUBFormatting(ROOT_FOLDER / "data/cub_dataset").build_metadata_csv_from_raw_data()
