#%%
from src.abo import ABO
from pathlib import Path
from torchvision import transforms
import numpy as np
import pandas as pd
import seaborn

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
# %%
color_class_grouped_df = pd.DataFrame(
    dataset.data.groupby(["label", "en_color"])["index"].count()
)
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
