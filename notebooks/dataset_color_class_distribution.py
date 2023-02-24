#%%
from src.abo import ABO
from pathlib import Path
from torchvision import transforms
import numpy as np
import pandas as pd
import seaborn

from src.cub import CUB

#%%
root = Path("data/abo_dataset/images/small")
image_size = 112

dataset = ABO(
    root=root,
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    classes_json=Path("data/selected_and_matched_abo_classes.json"),
    colors_json=Path("data/selected_and_removed_colors.json"),
)
color_class_grouped_df = pd.DataFrame(
    dataset.data.groupby(["label", "en_color"])["index"].count()
)
#%%
root_cub = Path("data/cub_dataset/images")
dataset_cub = CUB(
    root=root,
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
)
color_class_grouped_df_cub = pd.DataFrame(
    dataset_cub.data.groupby(["class_id", "color"])["image_id"].count()
)
# %%
color_X_class_dict = dict(
    (color, [0] * len(dataset.class_names)) for color in np.unique(dataset.get_colors())
)
count_list = list(color_class_grouped_df["index"])
i = 0
for label, color in color_class_grouped_df.index:
    color_X_class_dict[color][label] = count_list[i]
    i += 1
color_X_class_list = []
for key in color_X_class_dict:
    color_X_class_list.append(color_X_class_dict[key])
# %%
nb_color_per_class = [0] * 64
for label, color in color_class_grouped_df.index:
    nb_color_per_class[label] += 1
nb_color_per_class_cub = [0] * 200
for label, color in color_class_grouped_df_cub.index:
    nb_color_per_class_cub[label - 1] += 1
import matplotlib.pyplot as plt

bins = np.linspace(0.5, 16.5, 16)
plt.hist(nb_color_per_class, bins, alpha=0.5, label="abo")
plt.hist(nb_color_per_class_cub, bins, alpha=0.5, label="cub")
plt.legend(loc="upper right")
plt.xlabel("number of color in one class")
plt.ylabel("number of classes")
plt.show()
#%%
ax = seaborn.heatmap(
    color_X_class_list, yticklabels=np.unique(dataset.get_colors()), vmax=8
)
ax.set(xlabel="class", ylabel="color")
ax.xaxis.tick_bottom()
ax.set_title("Samples count for each pair of (color, class) in the ABO dataset")
# %%
class_count = []
for color in np.unique(dataset.get_colors()):
    class_count.append(sum(map(lambda x: x >= 8, color_X_class_dict[color])))
sum_up_df = pd.DataFrame(
    np.transpose([np.unique(dataset.get_colors()), class_count]),
    columns=["color", "number of class with at least 8 samples"],
)
sum_up_df
# %%
color_X_class_array = np.array(color_X_class_list)
color_count = []
for label in range(len(color_X_class_array[0])):
    color_count.append(int(sum(map(lambda x: x >= 8, color_X_class_array[:, label]))))

sum_up_df = pd.DataFrame(
    np.transpose([np.arange(len(color_X_class_array[0])), color_count]),
    columns=["label", "number of colors with at least 8 samples"],
)
selected_classes_index = list(
    sum_up_df.loc[sum_up_df["number of colors with at least 8 samples"] >= 2]["label"]
)
# %%
selected_classes = [dataset.class_names[index] for index in selected_classes_index]
# %%
