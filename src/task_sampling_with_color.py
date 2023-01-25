from abc import abstractmethod
from typing import List, Iterator, Tuple, Union
import torch
from torch import Tensor
import random
import pandas as pd

from easyfsl.samplers.task_sampler import TaskSampler
from src.abo import ABO
from src.cub import CUB


class TaskSamplerWithColor(TaskSampler):
    @abstractmethod
    def __init__(
        self,
        dataset: Union[ABO, CUB],
        n_query: int,
        n_tasks: int,
    ):
        """
        Args:
            dataset (FewShotDataset): dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_query (int): number of query images for each class in one task, works for 16 now
            n_tasks (int): number of tasks to sample
        """
        # TODO (21/11): makes n_query used, i.e. one can choose more than 16
        self.n_way = 2
        self.n_shot = 1
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.n_colors = 2

        self.items_df = pd.DataFrame(
            {
                "item": [i for i in range(len(dataset))],
                "label": dataset.get_labels(),
                "color": dataset.get_colors(),
                "img_path": dataset.get_img_path(),
            }
        )
        self.colors_list = list(self.items_df["color"].unique())
        self.class_list = list(self.items_df["label"].unique())

    @abstractmethod
    def __iter__(self) -> Iterator[List[int]]:
        pass

    @abstractmethod
    def populate_category(self) -> List:
        pass

    @abstractmethod
    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, int, str, str]]
    ) -> Tuple[Tensor, Tensor, List[str], Tensor, Tensor, List[str], List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
                - the color of this image
        Returns:
            tuple(Tensor, Tensor, List[str], Tensor, Tensor, List[str], List[int]): respectively:
                - support images,
                - their labels,
                - their colors,
                - query images,
                - their labels,
                - their colors,
                - the dataset class ids of the class sampled in the episode
        """
        true_class_ids = list({x[1] for x in input_data})
        colors_list = list(x[2] for x in input_data)
        img_path_list = list(x[3] for x in input_data)

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()
        support_colors = [colors_list[0], colors_list[self.n_query + self.n_shot]]
        query_colors = (
            colors_list[1 : self.n_query + 1]
            + colors_list[self.n_query + self.n_shot + 1 :]
        )
        support_img_path = [img_path_list[0], img_path_list[self.n_query + self.n_shot]]
        query_img_path = (
            img_path_list[1 : self.n_query + 1]
            + img_path_list[self.n_query + self.n_shot + 1 :]
        )

        return (
            support_images,
            support_labels,
            support_colors,
            support_img_path,
            query_images,
            query_labels,
            query_colors,
            query_img_path,
            true_class_ids,
        )


class ColorAwareTaskSampler(TaskSamplerWithColor):
    """
    Samples batches in the shape of one-shot classification tasks, and creating
    diagonal support set according to colors. At each iteration, it will sample
    2 classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: Union[ABO, CUB],
        n_query: int,
        n_tasks: int,
    ):
        super().__init__(dataset, n_query, n_tasks)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            # color A choice
            color_A = random.sample(self.colors_list, 1)[0]
            # classes with at least 8 elements of color A, and choice of one
            possible_classes = list(
                self.items_df.loc[self.items_df["color"] == color_A]
                .groupby(["label"])["color"]
                .count()
                .reset_index(name="count")
                .loc[lambda df: df["count"] >= 8]["label"]
            )
            class_1 = random.sample(possible_classes, 1)[0]
            # class 2 and color B choice
            possible_classes_with_enough_colors = (
                self.items_df.loc[
                    (self.items_df["label"].isin(possible_classes))
                    & (self.items_df["color"] != color_A)
                ]
                .groupby(["color", "label"])["label"]
                .count()
                .reset_index(name="count")
            )
            class_2 = random.sample(
                list(
                    possible_classes_with_enough_colors.loc[
                        (possible_classes_with_enough_colors["count"] >= 8)
                        & (possible_classes_with_enough_colors["label"] != class_1)
                    ]["label"]
                ),
                1,
            )[0]
            color_B = random.sample(
                list(
                    possible_classes_with_enough_colors.loc[
                        (possible_classes_with_enough_colors["label"] == class_2)
                        & (possible_classes_with_enough_colors["count"] >= 8)
                        & (possible_classes_with_enough_colors["color"] != color_A)
                    ]["color"]
                ),
                1,
            )[0]
            # sampling
            items_A_1 = self.populate_category(class_1, color_A, 9)
            items_B_1 = self.populate_category(class_1, color_B, 8)
            items_B_2 = self.populate_category(class_2, color_B, 9)
            items_A_2 = self.populate_category(class_2, color_A, 8)
            yield items_A_1 + items_B_1 + items_B_2 + items_A_2

    def populate_category(self, label: str, color: str, amount: int) -> List:
        items_label_class = list(
            self.items_df.loc[
                (self.items_df["color"] == color) & (self.items_df["label"] == label)
            ]["item"]
        )
        if len(items_label_class) >= amount:
            sampled_items = random.sample(items_label_class, amount)
        else:
            sampled_items = items_label_class + random.sample(
                list(
                    self.items_df.loc[
                        (self.items_df["label"] == label)
                        & (self.items_df["color"] != color)
                    ]["item"]
                ),
                amount - len(items_label_class),
            )
        return sampled_items


class NonColorAwareTaskSampler(TaskSamplerWithColor):
    """
    Samples batches in the shape of one-shot classification tasks. At each
    iteration, it will sample 2 classes, and then sample support and query
    images from these classes.
    """

    def __init__(
        self,
        dataset: Union[ABO, CUB],
        n_query: int,
        n_tasks: int,
    ):
        super().__init__(dataset, n_query, n_tasks)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            selected_classes = random.sample(self.class_list, 2)
            items_1 = self.populate_category(selected_classes[0])
            items_2 = self.populate_category(selected_classes[1])
            yield items_1 + items_2

    def populate_category(self, label: str) -> List:
        items_from_label = list(
            self.items_df.loc[(self.items_df["label"] == label)]["item"]
        )
        return random.sample(items_from_label, self.n_query + self.n_shot)
