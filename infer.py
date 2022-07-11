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
images = image_list

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

# fig = plt.figure(tight_layout=True)
# axes = fig.subplots(1, 2)
num = len(image_list)
fig, axes = plt.subplots(num, 2)

# select a sample from the batch
for index, (orig, result_image) in enumerate(zip(image_list, logits)):
    # permute to match the desired memory format
    result_image = result_image.permute(1, 2, 0).detach().numpy()
    # plt.imshow(result_image)
    # plt.show()
    axes[index, 0].imshow(orig)
    axes[index, 1].imshow(result_image)
plt.show()
