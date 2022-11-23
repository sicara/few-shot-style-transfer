from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import resnet18

from easyfsl.utils import plot_images
from easyfsl.methods.prototypical_networks import PrototypicalNetworks

from src.abo import ABO
from src.task_sampling_with_color import ColorAwareTaskSampler, NonColorAwareTaskSampler
from src.few_shot_classifier import EvaluatorFewShotClassifierWColor

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n-tasks",
    "--task_number",
    default=100,
    type=int,
    help="Number of few-shot tasks to do",
)
parser.add_argument(
    "-color",
    "--color_aware",
    default=False,
    type=bool,
    help="Whether or not you want to build tasks knowing the colors",
)
parser.add_argument(
    "-style",
    "--style_transfer_augmentation",
    default=False,
    type=bool,
    help="Whether or not you want to augment the support sets with style transfer",
)
args = parser.parse_args()
N_TASKS = args.task_number

ROOT = Path("data/abo_dataset/images/small")
IMAGE_SIZE = 112
N_WAY = 2  # Number of classes in a task
N_SHOT = 1  # Number of images per class in the support set
N_QUERY = 16  # Number of images per class in the query set

dataset = ABO(
    root=ROOT,
    classes_json=Path("data/selected_and_matched_abo_classes.json"),
    colors_json=Path("data/selected_and_removed_colors.json"),
)

if args.color_aware:
    test_sampler = ColorAwareTaskSampler(dataset, n_query=N_QUERY, n_tasks=N_TASKS)
else:
    test_sampler = NonColorAwareTaskSampler(dataset, n_query=N_QUERY, n_tasks=N_TASKS)

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
if args.style_transfer_augmentation:
    classified_dataset = EvaluatorFewShotClassifierWColor(few_shot_model).evaluate(
        test_loader, style_transfer_augmentation=True
    )
else:
    classified_dataset = EvaluatorFewShotClassifierWColor(few_shot_model).evaluate(
        test_loader, style_transfer_augmentation=False
    )
