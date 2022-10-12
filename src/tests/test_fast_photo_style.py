from src.style_transfer.fast_photo_style import FastPhotoStyle
from src.config import ROOT_FOLDER
from PIL import Image, ImageChops
from torchvision import transforms


class TestPhotoStyleTransfer:
    @staticmethod
    def test_photo_style_transfer_outputs_expected_image_but_not_save():
        content_image = transforms.ToTensor()(Image.open(ROOT_FOLDER / "src/style_transfer/examples" / "content1.png"))
        style_image = transforms.ToTensor()(Image.open(ROOT_FOLDER / "src/style_transfer/examples" / "style1.png"))

        img = FastPhotoStyle().photo_style_transfer(content_image, style_image)
        try:
            saved_img = Image.open(ROOT_FOLDER / "src/style_transfer/examples" / "output1.png")
        except:
            saved_img = None
        assert img is not None
        assert saved_img is None
        assert not ImageChops.difference(
            img, Image.open(ROOT_FOLDER / "src/style_transfer/examples" / "content1-style1.png")
        ).getbbox()
