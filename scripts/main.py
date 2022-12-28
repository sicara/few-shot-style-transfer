from pathlib import Path
import typer
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from easyfsl.utils import plot_images
from easyfsl.methods.prototypical_networks import PrototypicalNetworks

from src.abo import ABO
from src.evaluator_utils import (
    compute_accuracy_for_samples_with_same_color_as_class_representative,
    compute_accuracy_for_samples_with_same_color_as_no_class_representative,
    compute_accuracy_for_samples_with_same_color_as_other_class_representative,
)
from src.task_sampling_with_color import ColorAwareTaskSampler, NonColorAwareTaskSampler
from src.few_shot_classifier import EvaluatorFewShotClassifierWColor


def main(
    number_of_tasks: int = 100,
    color_aware: bool = False,
    style_transfer_augmentation: bool = False,
    save_results: bool = True,
):
    """Inference script for one-shot two-way image classification

    Args:
        number_of_tasks (int, optional): Number of few-shot tasks to do. Defaults to 100.
        color_aware (bool, optional): Whether or not you want to build tasks knowing the colors. Defaults to False.
        style_transfer_augmentation (bool, optional): Whether or not you want to augment the support sets with style transfer. Defaults to False.
        save_results (bool, optional): Whether or not you want to save the results as a csv. Defaults to True.

    """
    start_time = time.time()
    root = Path("data/abo_dataset/images/small")
    image_size = 112
    n_query = 16  # Number of images per class in the query set
    message = ""
    dataset = ABO(
        root=root,
        transform=transforms.Compose(
            [
                transforms.Pad(256, fill=255),
                transforms.CenterCrop(256),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        ),
        classes_json=Path("data/selected_and_matched_abo_classes.json"),
        colors_json=Path("data/selected_and_removed_colors.json"),
    )

    if color_aware:
        test_sampler = ColorAwareTaskSampler(
            dataset, n_query=n_query, n_tasks=number_of_tasks
        )
        print("--Color Task Sampler used")
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

    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    few_shot_model = PrototypicalNetworks(convolutional_network).cuda()
    classified_dataset = EvaluatorFewShotClassifierWColor(few_shot_model).evaluate(
        test_loader, style_transfer_augmentation=style_transfer_augmentation
    )
    if style_transfer_augmentation:
        print("--Style transfer augmented support sets")
        message += "style_"

    if save_results:
        classified_dataset.to_csv(
            "exp_results/exp_"
            + str(number_of_tasks)
            + "_"
            + message
            + datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
            + ".csv"
        )
    print("Execution time: ", round(time.time() - start_time, 2), "s")
    print(
        "Accuracy for samples with same color as class representative: ",
        compute_accuracy_for_samples_with_same_color_as_class_representative(
            classified_dataset
        ),
    )
    print(
        "Accuracy for samples with same color as other class representative: ",
        compute_accuracy_for_samples_with_same_color_as_other_class_representative(
            classified_dataset
        ),
    )
    print(
        "Accuracy for samples with same color as none of the class representative: ",
        compute_accuracy_for_samples_with_same_color_as_no_class_representative(
            classified_dataset
        ),
    )


if __name__ == "__main__":
    typer.run(main)
