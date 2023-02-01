import torch
import time
from loguru import logger
from src.style_transfer.utils.train_splice import train_model

def save_result(out_img):
  torch.save(out_img, "src/style_transfer/outputs/image_style_transformed2")

if __name__ == "__main__":
    start_time = time.time()
    train_model("src/style_transfer/examples/", save_result)
    logger.info(f"Execution time: {round(time.time() - start_time, 2)} s")
