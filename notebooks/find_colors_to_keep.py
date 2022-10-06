#%%
import pandas as pd
import json
from matplotlib import pyplot as plt
from src.config import ROOT_FOLDER

abo_data_df = pd.read_csv(ROOT_FOLDER / "data/gathered_abo_data.csv")
abo_data_df["en_color"] = abo_data_df["en_color"].astype(str)
#%%
color_count_df = pd.DataFrame(
    {"en_color": abo_data_df["en_color"].value_counts().index, "count": abo_data_df["en_color"].value_counts().values}
)
print("Number of different colors:", len(color_count_df))
color_count_df[color_count_df["count"] > 5]
# %%
threshold = 10
color_count_df[color_count_df["count"] > threshold].plot.bar(
    x="en_color",
    y="count",
    rot=90,
    figsize=(16, 2),
    title="Number of samples for each color (min " + str(threshold) + ")",
)
plt.show()
# %%
with open(ROOT_FOLDER / "data/selected_and_removed_colors.json") as json_file:
    selected_colors = json.load(json_file)["selected"]
with open(ROOT_FOLDER / "data/selected_and_matched_abo_classes.json") as json_file:
    selected_classes = json.load(json_file)["selected"]
# %%
final_abo_data = abo_data_df[
    (abo_data_df.en_color.isin(selected_colors)) & (abo_data_df.product_type.isin(selected_classes))
]
final_data_count_df = pd.DataFrame(
    {
        "product_type": final_abo_data["product_type"].value_counts().index,
        "count": final_abo_data["product_type"].value_counts().values,
    }
)
print("Number of classes with at least 17 samples:", len(final_data_count_df[final_data_count_df["count"] > 16]))
# %%
classes_with_few_samples = list(final_data_count_df[final_data_count_df["count"] < 17]["product_type"])
selected_classes = list(final_data_count_df[final_data_count_df["count"] > 16]["product_type"])

# %%
