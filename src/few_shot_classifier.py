import torch
from tqdm import tqdm
from torch import nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from src.style_transfer.fast_photo_style import FastPhotoStyle
from src.config import ROOT_FOLDER


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat([z_support[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


class FewShotClassifier:
    def __init__(self) -> None:
        self.few_shot_model = self.model_init()

    @staticmethod
    def model_init():
        convolutional_network = resnet18(pretrained=True)
        convolutional_network.fc = nn.Flatten()
        return PrototypicalNetworks(convolutional_network).cuda()

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
                self.few_shot_model(support_images.cuda(), support_labels.cuda(), query_images.cuda()).detach().data,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels)

    def evaluate(self, data_loader: DataLoader, style_transfer_augmentation: bool = False):
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
                correct, total = self.evaluate_on_one_task(support_images, support_labels, query_images, query_labels)

                total_predictions += total
                correct_predictions += correct

        print(
            f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
        )
