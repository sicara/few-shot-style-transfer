#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.evaluator_utils import (
    compute_accuracy_for_one_task,
    compute_task_color_similarity,
)

N_WAY = 2  # Number of classes in a task
N_SHOT = 1  # Number of images per class in the support set
N_QUERY = 16  # Number of images per class in the query set
N_EVALUATION_TASKS = 1000
#%%
prediction = pd.read_csv("exp_results/exp_1000_abo_05:01:2023_15:11:41.csv").drop(
    "Unnamed: 0", axis=1
)
#%%
task_accuracy_list = [
    compute_accuracy_for_one_task(prediction, task_id)
    for task_id in range(N_EVALUATION_TASKS)
]
task_color_similarity_list = np.array(
    [
        compute_task_color_similarity(prediction, task_id)
        for task_id in range(N_EVALUATION_TASKS)
    ]
)
metrics_per_task = (
    pd.DataFrame(
        list(
            zip(
                [i for i in range(N_EVALUATION_TASKS)],
                task_accuracy_list,
                task_color_similarity_list,
            )
        ),
        columns=["task_id", "accuracy", "color_similarity"],
    )
    .sort_values(by=["accuracy"], ascending=True)
    .reset_index(drop=True)
)
metrics_per_task = metrics_per_task.assign(
    smooth_color_similarity=lambda df: df.color_similarity.rolling(window=200).mean()
).dropna()
#%%
# create figure and axis objects with subplots()
fig, ax = plt.subplots()
# make a plot
ax.plot(metrics_per_task["accuracy"], color="red", marker="o")
# set x-axis label
ax.set_xlabel("task_id", fontsize=10)
# set y-axis label
ax.set_ylabel("accuracy", color="red", fontsize=10)
ax.set_ylim((0, 100))
# twin object for two different y-axis on the sample plot
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(metrics_per_task["smooth_color_similarity"], color="blue", marker="o")
ax2.set_ylabel("color similarity index", color="blue", fontsize=10)
plt.title(
    f"Accuracy for each task, for {N_WAY}-way {N_SHOT}-shot learning and {N_QUERY} query samples"
)
plt.show()
# %%
