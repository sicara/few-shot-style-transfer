#%% imports
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from torchvision.models import resnet18

from easyfsl.utils import plot_images
from easyfsl.methods.prototypical_networks import PrototypicalNetworks
from easyfsl.methods.tim import TIM
from src.few_shot_classifier import EvaluatorFewShotClassifierWColor
from src.config import ROOT_FOLDER
from src.evaluator_utils import compute_accuracy_for_one_task
import matplotlib.pyplot as plt
from src.basic_data_augmentation import BasicDataAugmentation
from src.style_transfer.fast_photo_style import FastPhotoStyle

IMAGE_FOLDER_PATH = "data/abo_dataset/images/small"
#%% Load exp result from csv
prediction = pd.read_csv(
    "/home/sicara/R&D/few-shot-style-transfer/exp_results/exp_100_abo_finetune_rotation_07:02:2023_11:22:07.csv"
).drop("Unnamed: 0", axis=1)
#%% find task with the lower accuracy
min_accuracy = 100
min_accuracy_index = -1
task_id_to_ignore = []  # [2, 21, 53, 64]
for task_id in prediction["task_id"].unique():
    temp = compute_accuracy_for_one_task(prediction, task_id)
    if temp < min_accuracy and task_id not in task_id_to_ignore:
        min_accuracy = temp
        min_accuracy_index = task_id
min_accuracy, min_accuracy_index
# %% get image path to reconstitute task
min_acc_task = prediction[prediction["task_id"] == 2]
support_img_path = (
    min_acc_task["support_set_0_img_path"].unique().tolist()
    + min_acc_task["support_set_1_img_path"].unique().tolist()
)
query_img_path = min_acc_task["img_path"].tolist()
#%% transform
to_tensor = transforms.Compose(
    [
        transforms.Pad(256, fill=255),
        transforms.CenterCrop(256),
        transforms.Resize(112),
        # transforms.RandomSolarize(0.5, p=1),
        # transforms.Grayscale(num_output_channels=3),
        # transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
    ]
)
to_tensor2 = transforms.Compose(
    [
        # transforms.CenterCrop(256),
        transforms.Resize((256, 256)),
        # transforms.RandomSolarize(0.5, p=1),
        # transforms.Grayscale(num_output_channels=3),
        # transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
    ]
)
#%% build custom dataset
support_img_path = [
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/green_chair_2.jpeg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/red_lamp_2.jpg",
]
query_img_path = [
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/black_chair_1.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/black_chair_2.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/black_chair_3.jpeg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/black_chair_4.jpeg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/green_chair_1.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/red_chair_2.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/green_chair_3.png",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/green_chair_4.jpeg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/green_chair_5.jpeg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/green_chair_6.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/red_chair_7.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/red_chair_5.jpeg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/red_chair_3.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/red_chair_4.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/red_chair_1.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/chair/red_chair_6.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/black_lamp_1.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/black_lamp_2.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/black_lamp_3.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/black_lamp_4.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/greeen_lamp_6.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/greeen_lamp_7.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/greeen_lamp_2.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/greeen_lamp_3.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/greeen_lamp_1.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/greeen_lamp_5.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/red_lamp_1.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/greeen_lamp_4.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/red_lamp_3.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/red_lamp_4.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/red_lamp_5.jpg",
    "/home/sicara/R&D/few-shot-style-transfer/data/custom_tasks/lamp/red_lamp_6.jpg",
]
support_images = []
query_images = []
for path in support_img_path:
    support_images.append(to_tensor(Image.open(path).convert("RGB")))
for path in query_img_path:
    query_images.append(to_tensor2(Image.open(path).convert("RGB")))
query_labels = torch.Tensor([0] * 16 + [1] * 16)
support_labels = torch.Tensor([0, 1])
query_images = torch.stack(query_images)
support_images = torch.stack(support_images)
#%% exchange support and query images
index_to_change = -12
temp = query_img_path[index_to_change]
query_img_path[index_to_change] = support_img_path[1]
support_img_path[1] = temp
#%% build task set
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
plot_images(support_images, title="", images_per_row=2)
plot_images(query_images, title="", images_per_row=8)
#%%
support_images, support_labels = BasicDataAugmentation("rotation").augment_support_set(
    support_images, support_labels
)
#%%
support_images, support_labels = FastPhotoStyle().augment_support_set(
    support_images, support_labels
)
#%%
plot_images(support_images, title="", images_per_row=2)
# %% visualize single image
Image.open(ROOT_FOLDER / IMAGE_FOLDER_PATH / "71/71199226.jpg").convert("RGB")
#%% define few shot model
convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = torch.nn.Flatten()
few_shot_model = TIM(convolutional_network, use_softmax=True).cuda()
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
#%% hist for prediction values of both classes (ex, abo 100, non-color aware, task id = 2)
l = [
    [-29.8940, -37.0054],
    [-30.6664, -34.3289],
    [-31.2647, -31.5017],
    [-29.4143, -33.0733],
    [-38.3203, -40.6224],
    [-25.7663, -30.7432],
    [-32.5462, -31.7854],
    [-27.5979, -32.4402],
    [-23.4712, -31.9282],
    [-28.5722, -31.3800],
    [-19.7518, -28.0787],
    [-25.5192, -32.6616],
    [-27.2403, -32.2408],
    [-26.0723, -34.8220],
    [-34.8659, -37.2368],
    [-26.9919, -33.5754],
    [-21.1100, -25.9459],
    [-28.4668, -30.0995],
    [-25.6552, -28.0215],
    [-22.9659, -28.7552],
    [-21.5565, -26.6932],
    [-22.8221, -28.6116],
    [-29.4652, -30.4940],
    [-24.1392, -24.3844],
    [-22.6342, -27.4754],
    [-22.4394, -28.3432],
    [-27.2251, -30.5400],
    [-23.2281, -26.6602],
    [-28.9033, -31.1441],
    [-26.7413, -29.4832],
    [-22.8577, -27.6786],
    [-26.6739, -27.9430],
]
lb = list(map(list, zip(*l)))
fig, ax = plt.subplots()
ax.hist(lb[0], 10, None, ec="red", fc="none", lw=1.5, histtype="step", label="class 0")
ax.hist(
    lb[1], 10, None, ec="green", fc="none", lw=1.5, histtype="step", label="class 1"
)
ax.legend(loc="upper left")
plt.show()

#%%
lb_0_n = [(float(i) - min(lb[0])) / (max(lb[0]) - min(lb[0])) for i in lb[0]]
lb_1_n = [(float(i) - min(lb[1])) / (max(lb[1]) - min(lb[1])) for i in lb[1]]
fig, ax = plt.subplots()
ax.hist(lb_0_n, 10, None, ec="red", fc="none", lw=1.5, histtype="step", label="class 0")
ax.hist(
    lb_1_n, 10, None, ec="green", fc="none", lw=1.5, histtype="step", label="class 1"
)
ax.legend(loc="upper left")
plt.show()
# %%
pred_n = []
for elem0, elem1 in zip(lb_0_n, lb_1_n):
    if elem0 < elem1:
        pred_n.append(1)
    else:
        pred_n.append(0)
pred_n = torch.Tensor(pred_n).cuda()
correct = (pred_n == query_labels.cuda()).sum().item()
total = len(query_labels)
print(100 * correct / total)
# %%
