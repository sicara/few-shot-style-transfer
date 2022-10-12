from src.style_transfer.fast_photo_style import FastPhotoStyle
from src.config import ROOT_FOLDER
from PIL import Image
from torchvision import transforms

content_image = transforms.ToTensor()(Image.open(ROOT_FOLDER / "src/style_transfer/examples" / "content1.png"))
style_image = transforms.ToTensor()(Image.open(ROOT_FOLDER / "src/style_transfer/examples" / "style1.png"))

img = FastPhotoStyle().photo_style_transfer(
    content_image, style_image, save_output_path=ROOT_FOLDER / "src/style_transfer/examples/output1.png"
)
