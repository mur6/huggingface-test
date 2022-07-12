# import cv2
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from src.utils import get_images

image_list = get_images()
images = image_list

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
# model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")

model_dir = sys.argv[1]
save_file_name = sys.argv[2]
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
model.eval()
inputs = feature_extractor_inference(images=images, return_tensors="pt")
outputs = model(**inputs)
# shape (batch_size, num_labels, height, width)

logits = outputs.logits

print(f"outputs: {type(outputs)}")
print(f"outputs.logis: {logits.shape}")

torch.save(logits, save_file_name)
