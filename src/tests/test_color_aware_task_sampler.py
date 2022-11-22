from pathlib import Path
import random
from torchvision import transforms

from src.task_sampling_with_color import ColorAwareTaskSampler
from src.abo import ABO

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


class TestIter:
    @staticmethod
    def test_total_number_of_samples():
        color_aware_task_sampler = ColorAwareTaskSampler(
            ABO_DATASET, n_query=16, n_tasks=100
        )
        random.seed(1)
        test_list = []
        for _ in range(5):
            test_list.append(len(next(iter(color_aware_task_sampler))))
        assert test_list == [34 for _ in range(5)]

    @staticmethod
    def test_min_number_of_colors():
        color_aware_task_sampler = ColorAwareTaskSampler(
            ABO_DATASET, n_query=16, n_tasks=100
        )
        random.seed(1)
        test_list = []
        for _ in range(5):
            task_data = next(iter(color_aware_task_sampler))
            colors_list = []
            for sample in task_data:
                color = ABO_DATASET.get_item_color(sample)
                if color not in colors_list:
                    colors_list.append(color)
            test_list.append(len(colors_list) >= 2)
        assert test_list == [True for _ in range(5)]

    @staticmethod
    def test_number_of_labels():
        color_aware_task_sampler = ColorAwareTaskSampler(
            ABO_DATASET, n_query=16, n_tasks=100
        )
        random.seed(1)
        test_list = []
        for _ in range(5):
            task_data = next(iter(color_aware_task_sampler))
            label_list = []
            for sample in task_data:
                label = ABO_DATASET.get_item_label(sample)
                if label not in label_list:
                    label_list.append(label)
            test_list.append(len(label_list))
        assert test_list == [2 for _ in range(5)]

    @staticmethod
    def test_color_A_in_class_1():
        color_aware_task_sampler = ColorAwareTaskSampler(
            ABO_DATASET, n_query=16, n_tasks=100
        )
        random.seed(1)
        test_list = []
        for _ in range(5):
            task_data = next(iter(color_aware_task_sampler))
            color_A = ABO_DATASET.get_item_color(task_data[0])
            class_1_color_A = 0
            for class_1_sample in task_data[1:17]:
                color = ABO_DATASET.get_item_color(class_1_sample)
                if color == color_A:
                    class_1_color_A += 1
            test_list.append(class_1_color_A >= 8)
        assert test_list == [True for _ in range(5)]

    @staticmethod
    def test_color_B_in_class_2():
        color_aware_task_sampler = ColorAwareTaskSampler(
            ABO_DATASET, n_query=16, n_tasks=100
        )
        random.seed(1)
        test_list = []
        for _ in range(5):
            task_data = next(iter(color_aware_task_sampler))
            color_B = ABO_DATASET.get_item_color(task_data[17])
            class_2_color_B = 0
            for class_2_sample in task_data[17:]:
                color = ABO_DATASET.get_item_color(class_2_sample)
                if color == color_B:
                    class_2_color_B += 1
            test_list.append(class_2_color_B >= 8)
        assert test_list == [True for _ in range(5)]

    @staticmethod
    def test_support_colors_are_different():
        color_aware_task_sampler = ColorAwareTaskSampler(
            ABO_DATASET, n_query=16, n_tasks=100
        )
        random.seed(1)
        test_list = []
        for _ in range(5):
            task_data = next(iter(color_aware_task_sampler))
            support_colors = [
                ABO_DATASET.get_item_color(task_data[0]),
                ABO_DATASET.get_item_color(task_data[17])[2],
            ]
            test_list.append(support_colors[0] != support_colors[1])
        assert test_list == [True for _ in range(5)]
