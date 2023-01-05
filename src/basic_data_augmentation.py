import torch
from torchvision import transforms
import random


class BasicDataAugmentation:
    def __init__(
        self,
        rotation: bool = True,
        deformation: bool = True,
        cropping: bool = True,
        flipping: bool = True,
        color_jiter: bool = True,
        solarize: bool = True,
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
        self.transforms_list = self.select_transforms(
            rotation, deformation, cropping, flipping, color_jiter, solarize, image_size
        )

    @staticmethod
    def select_transforms(
        rotation: bool,
        deformation: bool,
        cropping: bool,
        flipping: bool,
        color_jiter: bool,
        solarize: bool,
        image_size: int,
    ):
        transforms_list = []
        if rotation:
            transforms_list.append(transforms.RandomRotation((20, 340)))
        if deformation:
            transforms_list.append(transforms.RandomPerspective(p=1))
        if cropping:
            transforms_list.append(
                transforms.Compose(
                    [
                        transforms.RandomCrop(image_size / 2),
                        transforms.Resize(image_size),
                    ]
                )
            )
        if flipping:
            transforms_list.append(
                random.choice(
                    [
                        transforms.RandomHorizontalFlip(1),
                        transforms.RandomVerticalFlip(1),
                    ]
                )
            )
        if color_jiter:
            transforms_list.append(transforms.ColorJitter(0.5, 0.5, 0.5))
        if solarize:
            transforms_list.append(transforms.RandomSolarize(0.5, p=1))
        return transforms_list

    def image_augmentation(self, image: torch.Tensor):
        """
        Augment a given image by creating new images with each of the transforms available
        """
        augmented_images_from_image = [image]
        for transform in self.transforms_list:
            augmented_images_from_image.append(transform(image))
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
        augmented_support_images = []
        augmented_support_labels = []
        for img_id, img in enumerate(support_images):
            augmented_support_images += self.image_augmentation(img)
            augmented_support_labels += [support_labels[img_id]] * (
                len(self.transforms_list) + 1
            )
        return torch.stack(augmented_support_images), torch.stack(
            augmented_support_labels
        )
