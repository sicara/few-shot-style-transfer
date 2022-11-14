from typing import List, Iterator, Tuple
import torch
from torch import Tensor
import random
import numpy as np
from loguru import logger
import pandas as pd

from easyfsl.samplers.task_sampler import TaskSampler
from src.abo import ABO


class ColorAwareTaskSampler(TaskSampler):
    """
    Samples batches in the shape of few-shot classification tasks, and creating
    diagonal support set according to colors. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: ABO,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        """
        Args:
            dataset (FewShotDataset): dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way (int): number of classes in one task
            n_shot (int): number of support images for each class in one task
            n_query (int): number of query images for each class in one task
            n_tasks (int): number of tasks to sample
            n_colors (int, optional): Number of different colors used for one task (e.g. if equals 2, then one
                class is represented by color A and all other classes are represented by color B). Defaults to 2.
        """
        self.n_way = 2
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.n_colors = 2

        self.items_df = pd.DataFrame(
            {
                "item": [i for i in range(len(dataset))],
                "label": dataset.get_labels(),
                "color": dataset.get_colors(),
            }
        )

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            # HOW.1
            color_A = random.sample(list(self.items_df["color"].unique()), 1)[0]
            # HOW.2
            possible_classes = (
                self.items_df.loc[self.items_df["color"] == color_A]
                .groupby(["label"])["color"]
                .count()
                .reset_index(name="count")
            )
            possible_classes = list(
                possible_classes.loc[possible_classes["count"] >= 8]["label"]
            )
            class_1 = random.sample(possible_classes, 1)[0]
            # HOW.3
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
            # HOW.4
            items_A_1 = list(
                self.items_df.loc[
                    (self.items_df["color"] == color_A)
                    & (self.items_df["label"] == class_1)
                ]["item"]
            )
            if len(items_A_1) >= 9:
                items_A_1 = random.sample(items_A_1, 9)
            else:
                items_A_1 = items_A_1 + random.sample(
                    list(
                        self.items_df.loc[
                            (self.items_df["label"] == class_1)
                            & (self.items_df["color"] != color_A)
                        ]["item"]
                    ),
                    9 - len(items_A_1),
                )
            items_A_2 = random.sample(
                list(
                    self.items_df.loc[
                        (self.items_df["color"] == color_A)
                        & (self.items_df["label"] == class_2)
                    ]["item"]
                ),
                8,
            )
            items_B_2 = list(
                self.items_df.loc[
                    (self.items_df["color"] == color_B)
                    & (self.items_df["label"] == class_2)
                ]["item"]
            )
            if len(items_B_2) >= 9:
                items_B_2 = random.sample(items_B_2, 9)
            else:
                items_B_2 = items_B_2 + random.sample(
                    list(
                        self.items_df.loc[
                            (self.items_df["label"] == class_2)
                            & (self.items_df["color"] != color_B)
                        ]["item"]
                    ),
                    9 - len(items_B_2),
                )
            items_B_1 = list(
                self.items_df.loc[
                    (self.items_df["color"] == color_B)
                    & (self.items_df["label"] == class_1)
                ]["item"]
            )
            if len(items_B_1) >= 8:
                items_B_1 = random.sample(items_B_1, 8)
            else:
                items_B_1 = items_B_1 + random.sample(
                    list(
                        self.items_df.loc[
                            (self.items_df["label"] == class_1)
                            & (self.items_df["color"] != color_B)
                        ]["item"]
                    ),
                    8 - len(items_B_1),
                )
            yield items_A_1 + items_B_1 + items_B_2 + items_A_2

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, int, str]]
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
            tuple(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
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

        return (
            support_images,
            support_labels,
            support_colors,
            query_images,
            query_labels,
            query_colors,
            true_class_ids,
        )
