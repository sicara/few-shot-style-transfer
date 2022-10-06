#%%
from src.config import ROOT_FOLDER
import pandas as pd
import json
from matplotlib import pyplot as plt

abo_data_df = pd.read_csv(ROOT_FOLDER / "src/datasets/gathered_abo_data.csv")
abo_data_df["en_color"] = abo_data_df["en_color"].astype(str)
#%%
color_count_df = pd.DataFrame(
    {"en_color": abo_data_df["en_color"].value_counts().index, "count": abo_data_df["en_color"].value_counts().values}
)
print("Number of different colors:", len(color_count_df))
color_count_df[color_count_df["count"] > 5]
# %%
color_count_df[color_count_df["count"] > 100].plot.bar(
    x="en_color", y="count", rot=90, figsize=(16, 2), title="Number of samples for each color (min 100)"
)
plt.show()
# %%
with open(ROOT_FOLDER / "src/selected_and_removed_colors.json") as json_file:
    selected_colors = json.load(json_file)["selected"]
# %%
print(list(color_count_df.en_color))

# %%
