#%% imports
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from torchvision.models import resnet18

from easyfsl.utils import plot_images
from easyfsl.methods.prototypical_networks import PrototypicalNetworks
from src.few_shot_classifier import EvaluatorFewShotClassifierWColor
from src.config import ROOT_FOLDER
from src.evaluator_utils import compute_accuracy_for_one_task
import matplotlib.pyplot as plt

IMAGE_FOLDER_PATH = "data/abo_dataset/images/small"
#%% Load exp result from csv
prediction = pd.read_csv(
    "/home/sicara/R&D/few-shot-style-transfer/exp_results/exp_100_abo_31:01:2023_11:13:15.csv"
).drop("Unnamed: 0", axis=1)
#%% find task with the lower accuracy
min_accuracy = 100
min_accuracy_index = -1
task_id_to_ignore = [2, 21, 53, 64]
for task_id in prediction["task_id"].unique():
    temp = compute_accuracy_for_one_task(prediction, task_id)
    if temp < min_accuracy and task_id not in task_id_to_ignore:
        min_accuracy = temp
        min_accuracy_index = task_id
min_accuracy, min_accuracy_index
# %% build task set based on id, for the min accuracy task
min_acc_task = prediction[prediction["task_id"] == min_accuracy_index]
support_img_path = (
    min_acc_task["support_set_0_img_path"].unique().tolist()
    + min_acc_task["support_set_1_img_path"].unique().tolist()
)
query_img_path = min_acc_task["img_path"].tolist()
to_tensor = transforms.Compose(
    [
        transforms.Pad(256, fill=255),
        transforms.CenterCrop(256),
        transforms.Resize(112),
        transforms.ToTensor(),
    ]
)
support_images = []
query_images = []
for path in support_img_path:
    support_images.append(
        to_tensor(Image.open(ROOT_FOLDER / IMAGE_FOLDER_PATH / path).convert("RGB"))
    )
for path in query_img_path:
    query_images.append(
        to_tensor(Image.open(ROOT_FOLDER / IMAGE_FOLDER_PATH / path).convert("RGB"))
    )
query_labels = torch.Tensor(min_acc_task["true_label"].values.tolist())
support_labels = torch.Tensor(min_acc_task["true_label"].unique())
query_images = torch.stack(query_images)
support_images = torch.stack(support_images)
# %% plot support and query set
plot_images(support_images, title="support set", images_per_row=2)
plot_images(query_images, title="query set", images_per_row=8)
# %% visualize single image
Image.open(ROOT_FOLDER / IMAGE_FOLDER_PATH / "71/71199226.jpg").convert("RGB")
#%% define few shot model
convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = torch.nn.Flatten()
few_shot_model = PrototypicalNetworks(convolutional_network).cuda()
few_shot_model.eval()
#%% get prediction
prediction = EvaluatorFewShotClassifierWColor(few_shot_model).evaluate_on_one_task(
    support_images, support_labels, query_images
)
#%% compute accuracy
correct = (prediction == query_labels.cuda()).sum().item()
total = len(query_labels)
print(100 * correct / total)
# %% plot (tensor) image from file
sit = torch.load("/home/sicara/R&D/few-shot-style-transfer/sit_tensor.pt")
plt.imshow(sit[0].permute(1, 2, 0))
plt.show()
