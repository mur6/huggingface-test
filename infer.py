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
model.eval()
inputs = feature_extractor_inference(images=images, return_tensors="pt")
outputs = model(**inputs)
# shape (batch_size, num_labels, height, width)

logits = outputs.logits
print(logits.shape)


import matplotlib.pyplot as plt


# fig, axs = plt.subplots(1, 2, figsize=(20, 10))
# axs[0].imshow(img)
# axs[1].imshow(color_seg)
# plt.show()

# select a sample from the batch
for img in logits:
    # permute to match the desired memory format
    img = img.permute(1, 2, 0).detach().numpy()
    plt.imshow(img)
    plt.show()
