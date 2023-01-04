import torch
from torchvision import transforms
from tqdm import tqdm
import random


class BasicDataAugmentation:
    def __init__(
        self,
        rotation: bool = True,
        deformation: bool = True,
        cropping: bool = True,
        flipping: bool = True,
        color_jiter: bool = True,
        solarize: bool = False,
        image_size: int = 112,
    ):
        """
        Args:
            rotation: whether or not to apply rotation. Defaults to True.
            deformation: whether or not to apply perspective deformation. Defaults to True.
            cropping: whether or not to apply cropping. Defaults to True.
            flipping: whether or not to apply flipping. Defaults to True.
            color_jiter: whether or not to apply color jiter. Defaults to True.
            solarize: whether or not to apply solarize transformation. Defaults to True.
            image_size: the image size. Defaults to 112.
        """
        random.seed(1)
        torch.manual_seed(1)
        self.transforms_dict = {
            "rotation": [transforms.RandomRotation((20, 340)), rotation],
            "deformation": [transforms.RandomPerspective(p=1), deformation],
            "cropping": [
                transforms.Compose(
                    [
                        transforms.RandomCrop(image_size / 2),
                        transforms.Resize(image_size),
                    ]
                ),
                cropping,
            ],
            "flipping": [
                random.choice(
                    [
                        transforms.RandomHorizontalFlip(1),
                        transforms.RandomVerticalFlip(1),
                    ]
                ),
                flipping,
            ],
            "color_jiter": [transforms.ColorJitter(0.5, 0.5, 0.5), color_jiter],
            # "solarize": [transforms.RandomSolarize(0.5, p=1), solarize],
        }

    def image_augmentation(self, image: torch.Tensor):
        """
        Augment a given image by creating new images with each of the transforms available
        """
        transform_list = [
            transform_item[0]
            for transform_item in self.transforms_dict.values()
            if transform_item[1]
        ]
        augmented_images_from_image = None
        for transform in transform_list:
            if augmented_images_from_image is None:
                augmented_images_from_image = transform(image)[None, :]
            else:
                augmented_images_from_image = torch.cat(
                    (augmented_images_from_image, transform(image)[None, :]), 0
                )
        return augmented_images_from_image

    def augment_support_set(
        self, support_images: torch.Tensor, support_labels: torch.Tensor
    ):
        """
        Args:
            support_images: tensor containing the images of one task
            support_labels: tensor containing the labels for one task

        Returns:
            augmented_support_images: tensor containing the augmented support images for one task
            augmented_support_labels: tensor containing the augmented support labels for one task
        """
        augmented_support_images = support_images.detach().clone()
        augmented_support_labels = support_labels.detach().clone()

        number_of_transforms = len(
            [
                transform_item[0]
                for transform_item in self.transforms_dict.values()
                if transform_item[1]
            ]
        )
        for img_id, img in tqdm(
            enumerate(support_images), desc="Basic support set augmentation"
        ):
            augmented_support_images = torch.cat(
                (augmented_support_images, self.image_augmentation(img)), 0
            )
            augmented_support_labels = torch.cat(
                (
                    augmented_support_labels,
                    torch.tensor([support_labels[img_id]] * number_of_transforms),
                ),
                -1,
            )
        return augmented_support_images, augmented_support_labels
