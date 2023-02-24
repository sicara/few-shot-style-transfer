#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.evaluator_utils import (
    plot_task_accuracy_and_color_indexes,
)

N_WAY = 2  # Number of classes in a task
N_SHOT = 1  # Number of images per class in the support set
N_QUERY = 16  # Number of images per class in the query set
N_EVALUATION_TASKS = 1000
#%%
prediction = pd.read_csv(
    "/home/sicara/R&D/few-shot-style-transfer/exp_results/exp_1000_abo_05:01:2023_15:11:41.csv"
).drop("Unnamed: 0", axis=1)
#%%
plot_task_accuracy_and_color_indexes(prediction)
#%% FOR CUB
color_dict = {
    249: "blue",
    250: "brown",
    251: "iridescent",
    252: "purple",
    253: "rufous",
    254: "grey",
    255: "yellow",
    256: "olive",
    257: "green",
    258: "pink",
    259: "orange",
    260: "black",
    261: "white",
    262: "red",
    263: "buff",
}
prediction = prediction.replace({"color": color_dict})
# %% ONE TASK
colors_count = prediction.loc[prediction["task_id"] == 6]["color"].value_counts()
#%% ALL TASKS
colors_count = prediction["color"].value_counts()
#%%
color_list = colors_count.index.to_list()
# %%
accuracy_per_color = []
for color in color_list:
    correct = len(
        prediction[
            (prediction["color"] == color)
            & (prediction["true_label"] == prediction["predicted_label"])
        ]
    )
    total = colors_count[color]
    accuracy_per_color.append(100 * int(correct) / int(total))
accuracy_per_color = pd.Series(accuracy_per_color, index=color_list)
#%% for ABO
for id, color in enumerate(color_list):
    if color == "multicolored":
        color_list[id] = "cyan"
    if color == "flock":
        color_list[id] = "goldenrod"
#%% for CUB
for id, color in enumerate(color_list):
    if color == "iridescent":
        color_list[id] = "cyan"
    if color == "rufous":
        color_list[id] = "orangered"
    if color == "buff":
        color_list[id] = "peru"
#%%
colors_count.plot(kind="bar", color=color_list, edgecolor="black")
plt.show()
# %%
fig, ax1 = plt.subplots()
ax1.plot(color_list, accuracy_per_color.values, color="black")
ax1.set_ylabel("accuracy")
ax1.set_ylim(0, 100)
ax1.set_xticklabels(color_list, rotation=45)
ax2 = ax1.twinx()
ax2.bar(
    color_list,
    colors_count.values,
    width=0.5,
    alpha=0.8,
    color=color_list,
    edgecolor="black",
)
ax2.grid(False)
ax2.set_ylabel("nb")
plt.show()
# %%
from statistics import mean, stdev

nb_of_color_per_task = []
for task_id in range(N_EVALUATION_TASKS):
    nb_of_color_per_task.append(
        len(prediction.loc[prediction["task_id"] == task_id]["color"].unique())
    )
mean(nb_of_color_per_task), 1.96 * stdev(nb_of_color_per_task) / np.sqrt(
    len(nb_of_color_per_task)
), min(nb_of_color_per_task), max(nb_of_color_per_task)
# %%
