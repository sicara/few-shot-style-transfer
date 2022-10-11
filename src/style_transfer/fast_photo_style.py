from xmlrpc.client import Boolean
from src.config import ROOT_FOLDER
from pathlib import Path
from typing import Union
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

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
        p_wct.load_state_dict(
            torch.load(ROOT_FOLDER / self.root / "utils/photo_wct.pth")
        )
        if cuda_presence:
            p_wct.cuda(0)
        return p_wct

    def photo_style_transfer(
        self,
        content_photo: torch.Tensor,
        style_photo: torch.Tensor,
        save_not_return: bool = True,
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
            output_image_path=ROOT_FOLDER / self.root / "examples" / "example1.png",
            cuda=1,
            save_intermediate=False,
            no_post=False,
            save_not_return=save_not_return,
        )
        if img is not None:
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
        convert_img_to_tensor = transforms.ToTensor()

        augmented_support_images = support_images.detach().clone()
        augmented_support_labels = support_labels.detach().clone()
        N_WAY = len(support_images)

        for content_img_id in tqdm(range(N_WAY), desc="Support set augmentation"):
            for style_img_id in range(N_WAY):
                if content_img_id != style_img_id:
                    new_img = convert_img_to_tensor(
                        self.photo_style_transfer(
                            support_images[content_img_id],
                            support_images[style_img_id],
                            save_not_return=False,
                        )
                    )[None, :]
                    new_img = F.interpolate(
                        new_img,
                        size=(
                            augmented_support_images.shape[-1],
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
                            torch.tensor([augmented_support_labels[content_img_id]]),
                        ),
                        -1,
                    )
        return augmented_support_images, augmented_support_labels
