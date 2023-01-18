import torch
from torchvision import transforms

from src.constants import IMAGE_SIZE

AUGMENTATION = {
    "rotation": transforms.RandomRotation((20, 340), fill=255),
    "deformation": transforms.RandomPerspective(p=1, fill=255),
    "cropping": transforms.Compose(
        [
            transforms.RandomCrop(IMAGE_SIZE / 2),
            transforms.Resize(IMAGE_SIZE),
        ]
    ),
    "vertical_flipping": transforms.RandomVerticalFlip(1),
    "horizontal_flipping": transforms.RandomHorizontalFlip(1),
    "color_jiter": transforms.ColorJitter(0.5, 0.5, 0.5),
    "solarize": transforms.RandomSolarize(0.5, p=1),
}


class BasicDataAugmentation:
    def __init__(
        self,
        augmentations: str = "rotation,deformation,cropping,vertical_flipping,horizontal_flipping,color_jiter,solarize",
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
        self.transforms_list = [
            AUGMENTATION[augmentation_item]
            for augmentation_item in augmentations.split(",")
        ]

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
