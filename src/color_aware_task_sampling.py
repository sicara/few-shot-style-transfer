from typing import List, Iterator
import torch
import random
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
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        n_colors: int = 2,
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
        super().__init__(dataset, n_way, n_shot, n_query, n_tasks)
        if n_colors > n_way:
            logger.info(
                f"Only {n_way} different color(s) used, because we only have {n_way} class(es)"
            )
        self.n_colors = n_colors

        self.items_df = pd.DataFrame(
            {
                "item": [i for i in range(len(dataset))],
                "label": dataset.get_labels(),
                "color": dataset.get_colors(),
            }
        )

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            color_list = random.sample(
                list(set(self.dataset.get_colors())), self.n_colors
            )
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            list(
                                self.items_df.loc[
                                    (self.items_df["label"] == label)
                                    & (self.items_df["color"] in color_list),
                                    ["item"],
                                ]
                            ),
                            self.n_shot + self.n_query,
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(
                        list(self.items_df["label"].unique()), self.n_way
                    )
                ]
            ).tolist()
