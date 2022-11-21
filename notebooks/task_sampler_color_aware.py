#%%
from src.abo import ABO
from src.style_transfer.fast_photo_style import FastPhotoStyle
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from src.color_aware_task_sampling import ColorAwareTaskSampler
from easyfsl.utils import plot_images

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
# %%
