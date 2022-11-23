#%%
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from easyfsl.utils import plot_images
from easyfsl.methods.prototypical_networks import PrototypicalNetworks

from src.abo import ABO
from src.style_transfer.fast_photo_style import FastPhotoStyle
from src.task_sampling_with_color import ColorAwareTaskSampler
from src.few_shot_classifier import EvaluatorFewShotClassifierWColor

#%%
root = Path("data/abo_dataset/images/small")
image_size = 112

dataset = ABO(
    root=root,
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    classes_json=Path("data/selected_and_matched_abo_classes.json"),
    colors_json=Path("data/selected_and_removed_colors.json"),
)
N_QUERY = 16  # Number of images per class in the query set
N_EVALUATION_TASKS = 100
#%%
test_sampler = ColorAwareTaskSampler(
    dataset, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)
test_loader = DataLoader(
    dataset,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)
(
    example_support_images,
    example_support_labels,
    example_support_colors,
    example_query_images,
    example_query_labels,
    example_query_colors,
    example_true_class_ids,
) = next(iter(test_loader))
#%%
plot_images(
    example_support_images,
    "support images: " + example_support_colors[0] + " & " + example_support_colors[1],
    images_per_row=2,
)
plot_images(example_query_images, "query images", images_per_row=N_QUERY)

# %%
(
    augmented_support_images,
    augmented_support_labels,
) = FastPhotoStyle().augment_support_set(example_support_images, example_support_labels)
# %%
plot_images(augmented_support_images, "support images", images_per_row=4)
#%%
convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
few_shot_model = PrototypicalNetworks(convolutional_network).cuda()
classified_dataset = EvaluatorFewShotClassifierWColor(few_shot_model).evaluate(
    test_loader, style_transfer_augmentation=False
)
# %%
classified_dataset["same_color_as_support_img"] = classified_dataset.apply(
    lambda row: ((row.color == row.support_set_1_color) and (row.true_label == 0.0))
    or ((row.color == row.support_set_2_color) and (row.true_label == 1.0)),
    axis=1,
)
classified_dataset["same_color_as_other_support_img"] = classified_dataset.apply(
    lambda row: ((row.color == row.support_set_1_color) and (row.true_label == 1.0))
    or ((row.color == row.support_set_2_color) and (row.true_label == 0.0)),
    axis=1,
)
total_number = len(classified_dataset)
misclassified_number = len(
    classified_dataset.loc[
        classified_dataset["true_label"] != classified_dataset["predicted_label"]
    ]
)
misclassified_with_same_color_as_support = classified_dataset.loc[
    (classified_dataset["same_color_as_support_img"] == True)
    & (classified_dataset["true_label"] != classified_dataset["predicted_label"])
]
with_same_color_as_support = classified_dataset.loc[
    (classified_dataset["same_color_as_support_img"] == True)
]
misclassified_with_same_color_as_other_support = classified_dataset.loc[
    (classified_dataset["same_color_as_other_support_img"] == True)
    & (classified_dataset["true_label"] != classified_dataset["predicted_label"])
]
with_same_color_as_other_support = classified_dataset.loc[
    (classified_dataset["same_color_as_other_support_img"] == True)
]
print("total", total_number)
print("misclassified_number", misclassified_number)
print(
    "misclassified_with_same_color_as_support",
    len(misclassified_with_same_color_as_support),
    ": ",
    100
    * len(misclassified_with_same_color_as_support)
    / len(with_same_color_as_support),
    "%",
)
print(
    "misclassified_with_same_color_as_other_support",
    len(misclassified_with_same_color_as_other_support),
    ": ",
    100
    * len(misclassified_with_same_color_as_other_support)
    / len(with_same_color_as_other_support),
    "%",
)
# %%
