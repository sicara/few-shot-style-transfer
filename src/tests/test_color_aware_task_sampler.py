from src.color_aware_task_sampling import ColorAwareTaskSampler
from src.abo import ABO
from pathlib import Path
from torchvision import transforms
import pandas as pd

ABO_DATASET = ABO(
    root=Path("data/abo_dataset/images/small"),
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    classes_json=Path("data/selected_and_matched_abo_classes.json"),
    colors_json=Path("data/selected_and_removed_colors.json"),
)

COLOR_AWARE_TASK_SAMPLER = ColorAwareTaskSampler(ABO_DATASET, n_query=16, n_tasks=100)


class TestIter:
    @staticmethod
    def test_total_number_of_samples():
        assert len(next(iter(COLOR_AWARE_TASK_SAMPLER))) == 34

    @staticmethod
    def test_min_number_of_colors():
        task_data = next(iter(COLOR_AWARE_TASK_SAMPLER))
        colors_list = []
        for sample in task_data:
            color = ABO_DATASET.__getitem__(sample)[2]
            if color not in colors_list:
                colors_list.append(color)
        assert len(colors_list) >= 2

    @staticmethod
    def test_number_of_labels():
        task_data = next(iter(COLOR_AWARE_TASK_SAMPLER))
        label_list = []
        for sample in task_data:
            label = ABO_DATASET.__getitem__(sample)[1]
            if label not in label_list:
                label_list.append(label)
        assert len(label_list) == 2

    @staticmethod
    def test_support_colors_in_all_classes():
        task_data = next(iter(COLOR_AWARE_TASK_SAMPLER))
        support_colors = [
            ABO_DATASET.__getitem__(task_data[0])[2],
            ABO_DATASET.__getitem__(task_data[17])[2],
        ]
        class_1_colors = []
        for class_1_sample in task_data[1:17]:
            color = ABO_DATASET.__getitem__(class_1_sample)[2]
            if color not in class_1_colors and color in support_colors:
                class_1_colors.append(color)
        class_2_colors = []
        for class_2_sample in task_data[18:]:
            color = ABO_DATASET.__getitem__(class_2_sample)[2]
            if color not in class_2_colors and color in support_colors:
                class_2_colors.append(color)
        assert len(class_1_colors) == 2 and len(class_2_colors) == 2

    @staticmethod
    def test_support_colors_are_different():
        task_data = next(iter(COLOR_AWARE_TASK_SAMPLER))
        support_colors = [
            ABO_DATASET.__getitem__(task_data[0])[2],
            ABO_DATASET.__getitem__(task_data[17])[2],
        ]
        assert support_colors[0] != support_colors[1]
