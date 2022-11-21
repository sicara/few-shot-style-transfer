import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.style_transfer.fast_photo_style import FastPhotoStyle
from src.config import ROOT_FOLDER
from statistics import mean, stdev


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
        self.few_shot_model.process_support_set(
            support_images.cuda(), support_labels.cuda()
        )
        return (
            torch.max(
                self.few_shot_model(query_images.cuda()).detach().data,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels)

    def evaluate(
        self, data_loader: DataLoader, style_transfer_augmentation: bool = False
    ):
        accuracy = []

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
                    (
                        support_images,
                        support_labels,
                    ) = FastPhotoStyle().augment_support_set(
                        support_images, support_labels
                    )
                correct, total = self.evaluate_on_one_task(
                    support_images, support_labels, query_images, query_labels
                )
                accuracy.append(100 * correct / total)

        print(
            f"Model tested on {len(data_loader)} tasks. Accuracy: {mean(accuracy):.2f}% +/- {(1.96*stdev(accuracy)/np.sqrt(len(accuracy))):.2f}"
        )
