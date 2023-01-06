#%%
from src.style_transfer.fast_photo_style import FastPhotoStyle
import torchvision.transforms as transforms
from PIL import Image
from src.config import ROOT_FOLDER
transform = transforms.Compose([transforms.PILToTensor()])
#%%
chair = Image.open(ROOT_FOLDER / "src/style_transfer/examples/red_chair.png")
chair = transform(chair)
print(chair.shape)
pouf = Image.open(ROOT_FOLDER / "src/style_transfer/examples/content1.png")
pouf = transform(pouf)
print(pouf.shape)
meuble = Image.open(ROOT_FOLDER / "src/style_transfer/examples/style1.png")
meuble = transform(meuble)
print(meuble.shape)
# %%
FastPhotoStyle().photo_style_transfer(pouf, meuble, "test.png")
# %%
