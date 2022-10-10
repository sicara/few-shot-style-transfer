from src.style_transfer.fast_photo_style import FastPhotoStyle
from src.config import ROOT_FOLDER

img = FastPhotoStyle(ROOT_FOLDER / "src/style_transfer").photo_style_transfer(
    "content1.png", "style1.png", save_not_return=True
)

print("Style transferred image: ", img)
