#%%
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch

from easyfsl.utils import plot_images
from src.config import ROOT_FOLDER
from src.evaluator_utils import compute_accuracy_for_one_task

IMAGE_FOLDER_PATH = "data/abo_dataset/images/small"
#%%
prediction = pd.read_csv(
    "/home/sicara/R&D/few-shot-style-transfer/exp_results/exp_100_abo_25:01:2023_16:52:05.csv"
).drop("Unnamed: 0", axis=1)
#%%
min_accuracy = 100
min_accuracy_index = -1
task_id_to_ignore = [2, 21, 53, 64]
for task_id in prediction["task_id"].unique():
    temp = compute_accuracy_for_one_task(prediction, task_id)
    if temp < min_accuracy and task_id not in task_id_to_ignore:
        min_accuracy = temp
        min_accuracy_index = task_id
min_accuracy, min_accuracy_index
# %%
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
        transforms.PILToTensor(),
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

# %%
plot_images(torch.stack(support_images), title="support set", images_per_row=2)
plot_images(torch.stack(query_images), title="query set", images_per_row=8)
# %%
Image.open(ROOT_FOLDER / IMAGE_FOLDER_PATH / "71/71199226.jpg").convert("RGB")
# %%
