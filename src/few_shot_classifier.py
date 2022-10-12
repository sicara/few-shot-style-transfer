import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.style_transfer.fast_photo_style import FastPhotoStyle
from src.config import ROOT_FOLDER


class EvaluatorFewShotClassifier:
    def __init__(self, few_shot_classifier) -> None:
        self.few_shot_model = few_shot_classifier

    def evaluate_on_one_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> [int, int]:
        """
        Returns the number of correct predictions of query labels, and the total number of predictions.
        """
        return (
            torch.max(
                self.few_shot_model(
                    support_images.cuda(), support_labels.cuda(), query_images.cuda()
                )
                .detach()
                .data,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels)

    def evaluate(
        self, data_loader: DataLoader, style_transfer_augmentation: bool = False
    ):
        # We'll count everything and compute the ratio at the end
        total_predictions = 0
        correct_predictions = 0

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
        self.few_shot_model.eval()
        with torch.no_grad():
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                class_ids,
            ) in tqdm(enumerate(data_loader), total=len(data_loader)):
                if style_transfer_augmentation:
                    support_images, support_labels = FastPhotoStyle(
                        ROOT_FOLDER / "src/style_transfer"
                    ).augment_support_set(support_images, support_labels)
                correct, total = self.evaluate_on_one_task(
                    support_images, support_labels, query_images, query_labels
                )

                total_predictions += total
                correct_predictions += correct

        print(
            f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
        )
