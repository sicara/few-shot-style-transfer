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
    "/home/sicara/R&D/few-shot-style-transfer/exp_results/exp_10_cub_color_06:01:2023_17:31:42.csv"
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
