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
# %%
