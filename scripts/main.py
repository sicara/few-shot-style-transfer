from pathlib import Path
import random
import typer
from loguru import logger
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision import transforms
from torchvision.models import resnet18

from easyfsl.methods.prototypical_networks import PrototypicalNetworks
from easyfsl.methods.tim import TIM
from easyfsl.methods.finetune import Finetune

from src.abo import ABO
from src.cub import CUB
from src.evaluator_utils import (
    compute_accuracy_for_samples_with_same_color_as_class_representative,
    compute_accuracy_for_samples_with_same_color_as_no_class_representative,
    compute_accuracy_for_samples_with_same_color_as_other_class_representative,
)
from src.task_sampling_with_color import ColorAwareTaskSampler, NonColorAwareTaskSampler
from src.few_shot_classifier import EvaluatorFewShotClassifierWColor

FEW_SHOT_MODELS = {
    "prototypical": PrototypicalNetworks,
    "tim": TIM,
    "finetune": Finetune,
}
CONVOLUTIONAL_NETWORK = resnet18(pretrained=True)


def main(
    number_of_tasks: int = 100,
    color_aware: bool = False,
    few_shot_method: str = "prototypical",
    style_transfer_augmentation: bool = False,
    basic_augmentation: str = None,
    dataset_used: str = "abo",
    save_results: bool = True,
):
    """Inference script for one-shot two-way image classification

    Args:
        number_of_tasks (int, optional): Number of few-shot tasks to do. Defaults to 100.
        color_aware (bool, optional): Whether or not you want to build tasks knowing the colors. Defaults to False.
        few_shot_method (str, optional): The few-shot classifier used, either 'prototypical', 'tim', 'finetune'. Defaults to 'prototypical'.
        style_transfer_augmentation (bool, optional): Whether or not you want to augment the support sets with style transfer. Defaults to False.
        basic_augmentation (str, optional): The basic transforms used as support set augmentation, taken in the following
            'rotation,deformation,cropping,vertical_flipping,horizontal_flipping,color_jiter,solarize'. Defaults to None.
        dataset_used (str, optional): The dataset used, either 'abo' or 'cub'. Defaults to 'abo'.
        save_results (bool, optional): Whether or not you want to save the results as a csv. Defaults to True.

    """
    start_time = time.time()
    image_size = 112
    n_query = 16  # Number of images per class in the query set
    message = ""
    random.seed(1)
    torch.manual_seed(1)
    transform = transforms.Compose(
        [
            transforms.Pad(256, fill=255),
            transforms.CenterCrop(256),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    if dataset_used == "cub":
        root = Path("data/cub_dataset/images")
        dataset = CUB(root=root, transform=transform)
    elif dataset_used == "abo":
        root = Path("data/abo_dataset/images/small")
        dataset = ABO(
            root=root,
            transform=transform,
            classes_json=Path("data/selected_and_matched_abo_classes.json"),
            colors_json=Path("data/selected_and_removed_colors.json"),
        )
    else:
        raise ValueError("Invalid dataset input. Should be either 'abo' or 'cub'.")

    if color_aware:
        test_sampler = ColorAwareTaskSampler(
            dataset, n_query=n_query, n_tasks=number_of_tasks
        )
        logger.info("--Color Task Sampler used")
        message += "color_"
    else:
        test_sampler = NonColorAwareTaskSampler(
            dataset, n_query=n_query, n_tasks=number_of_tasks
        )

    test_loader = DataLoader(
        dataset,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    convolutional_network = CONVOLUTIONAL_NETWORK
    convolutional_network.fc = nn.Flatten()
    if few_shot_method in FEW_SHOT_MODELS:
        few_shot_model = FEW_SHOT_MODELS[few_shot_method](convolutional_network).cuda()
        logger.info(f"--{few_shot_method} model used")
        message += f"{few_shot_method}_"
    else:
        raise ValueError("Unknown few-shot method. Should be either 'prototypical' or 'tim' or 'finetune.")
    few_shot_model.eval()
    classified_dataset = EvaluatorFewShotClassifierWColor(few_shot_model).evaluate(
        test_loader,
        style_transfer_augmentation=style_transfer_augmentation,
        basic_augmentation=basic_augmentation,
    )
    if style_transfer_augmentation:
        logger.info("--Style transfer augmented support sets")
        message += "style_"
    if basic_augmentation is not None:
        logger.info(f"--Basic transforms ({basic_augmentation}) augmented support sets")
        for augmentation in basic_augmentation.split(","):
            message += f"{augmentation}_"

    if save_results:
        classified_dataset.to_csv(
            f"exp_results/"
            f"exp_{number_of_tasks}_{dataset_used}_{message}"
            f"{datetime.now().strftime('%d:%m:%Y_%H:%M:%S')}.csv"
        )
    logger.info(f"Execution time: {round(time.time() - start_time, 2)} s")
    (
        A_same,
        same_set,
    ) = compute_accuracy_for_samples_with_same_color_as_class_representative(
        classified_dataset
    )
    logger.success(f"A_same: {round(A_same, 2)}%, on {same_set} samples")
    (
        A_other,
        other_set,
    ) = compute_accuracy_for_samples_with_same_color_as_other_class_representative(
        classified_dataset
    )
    logger.success(f"A_other: {round(A_other,2)}%, on {other_set} samples")
    (
        A_none,
        none_set,
    ) = compute_accuracy_for_samples_with_same_color_as_no_class_representative(
        classified_dataset
    )
    logger.success(f"A_none: {round(A_none,2)}%, on {none_set} samples")
    print("-------------------------------------------------")


if __name__ == "__main__":
    typer.run(main)
