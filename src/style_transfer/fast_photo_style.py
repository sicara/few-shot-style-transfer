from xmlrpc.client import Boolean
from src.config import ROOT_FOLDER
from pathlib import Path
from typing import Union
import torch

from src.style_transfer.utils.photo_wct import PhotoWCT
from src.style_transfer.utils.photo_smooth import Propagator
from src.style_transfer.utils import process_stylization


class FastPhotoStyle:
    def __init__(self, root: Union[str, Path], cuda_presence=True):
        """
        Args:
            root: path to the root folder for style transfer
            cuda_presence (bool, optional): presence of cuda. Defaults to True.
        """
        self.root = Path(root)
        self.photo_wct_model = self.load_model_wct(cuda_presence)
        self.photo_propagator = Propagator()

    def load_model_wct(self, cuda_presence):
        p_wct = PhotoWCT()
        p_wct.load_state_dict(torch.load(ROOT_FOLDER / self.root / "utils/photo_wct.pth"))
        if cuda_presence:
            p_wct.cuda(0)
        return p_wct

    def photo_style_transfer(self, content_photo_path: Union[str, Path], style_photo_path: Union[str, Path]):
        process_stylization.stylization(
            stylization_module=self.photo_wct_model,
            smoothing_module=self.photo_propagator,
            content_image_path=ROOT_FOLDER / self.root / "examples" / Path(content_photo_path),
            style_image_path=ROOT_FOLDER / self.root / "examples" / Path(style_photo_path),
            content_seg_path=[],
            style_seg_path=[],
            output_image_path=ROOT_FOLDER
            / self.root
            / "examples"
            / Path(
                str(content_photo_path)[:-4] + "-" + str(style_photo_path)
            ),  # f2/f2ee3bf8.jpg and 62/62bc0963.jpg give f2/f2ee3bf8-62/62bc0963.jpg
            cuda=1,
            save_intermediate=False,
            no_post=False,
        )
