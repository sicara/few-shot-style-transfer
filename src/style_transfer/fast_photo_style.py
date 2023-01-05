from xmlrpc.client import Boolean
from src.config import ROOT_FOLDER
from pathlib import Path
from typing import Union, Optional
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

from src.style_transfer.utils.photo_wct import PhotoWCT
from src.style_transfer.utils.photo_smooth import Propagator
from src.style_transfer.utils import process_stylization


class FastPhotoStyle:
    def __init__(self, cuda_presence=True):
        """
        Args:
            root: path to the root folder for style transfer
            cuda_presence (bool, optional): presence of cuda. Defaults to True.
        """
        self.photo_wct_model = self.load_model_wct(cuda_presence)
        self.photo_propagator = Propagator()

    def load_model_wct(self, cuda_presence):
        p_wct = PhotoWCT()
        p_wct.load_state_dict(torch.load(ROOT_FOLDER / "models/photo_wct.pth"))
        if cuda_presence:
            p_wct.cuda(0)
        return p_wct

    def photo_style_transfer(
        self,
        content_photo: torch.Tensor,
        style_photo: torch.Tensor,
        save_output_path: Optional[Union[Path, str]] = None,
    ):
        """
        Args:
            content_photo (torch.Tensor): tensor representing the photo with content
            style_photo (torch.Tensor): tensor representing the photo with style
            save_not_return (bool, optional): if True the method saves the output image, if False the method returns the image as a PIL Image. Defaults to True.
        """
        img = process_stylization.stylization(
            stylization_module=self.photo_wct_model,
            smoothing_module=self.photo_propagator,
            content_image=transforms.ToPILImage()(content_photo).convert("RGB"),
            style_image=transforms.ToPILImage()(style_photo).convert("RGB"),
            content_seg_path=[],
            style_seg_path=[],
            output_image_path=(
                ROOT_FOLDER / save_output_path if save_output_path is not None else None
            ),
            cuda=1,
            save_intermediate=False,
            no_post=False,
        )
        return img

    def augment_support_set(
        self, support_images: torch.Tensor, support_labels: torch.Tensor
    ):
        """
        Args:
            support_images (torch.Tensor): tensor containing the images of one task
            support_labels (torch.Tensor): tensor containing the labels for one task
        Returns:
            augmented_support_images (torch.Tensor): tensor containing the augmented support images for one task
            augmented_support_labels (torch.Tensor): tensor containing the augmented support labels for one task
        """

        augmented_support_images = support_images.detach().clone()
        augmented_support_labels = support_labels.detach().clone()

        for content_img_id, content_img in enumerate(support_images):
            for style_img_id, style_img in enumerate(support_images):
                if content_img_id != style_img_id:
                    new_img = transforms.ToTensor()(
                        self.photo_style_transfer(
                            content_img,
                            style_img,
                        )
                    ).unsqueeze(0)
                    new_img = F.interpolate(
                        new_img,
                        size=(
                            augmented_support_images.shape[-2],
                            augmented_support_images.shape[-1],
                        ),
                    )
                    augmented_support_images = torch.cat(
                        (augmented_support_images, new_img),
                        0,
                    )
                    augmented_support_labels = torch.cat(
                        (
                            augmented_support_labels,
                            torch.tensor([support_labels[content_img_id]]),
                        ),
                        -1,
                    )
        return augmented_support_images, augmented_support_labels
