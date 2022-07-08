from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch

# import cv2
from pathlib import Path
from PIL import Image


def iter_pil_images():
    # samples_dir = sys.argv[1]
    samples_dir = "/Users/taichi.muraki/workspace/machine-learning/mur6-lightning-flash-test/data/samples"
    samples_dir = Path(samples_dir)
    sample_images = list(samples_dir.glob("*.jpeg"))
    for p in sample_images:
        image = Image.open(p)
        yield image


image_list = list(iter_pil_images())
images = image_list[0:3]

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
# model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
model = SegformerForSemanticSegmentation.from_pretrained("./models")

inputs = feature_extractor_inference(images=images, return_tensors="pt")
outputs = model(**inputs)
# shape (batch_size, num_labels, height, width)

logits = outputs.logits
print(logits.shape)
