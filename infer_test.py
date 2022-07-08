import os
import pandas as pd
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

from PIL import Image
import matplotlib.pyplot as plt
from torch import nn


# df = pd.read_csv('drone_dataset/class_dict_seg.csv')
# classes = df['name']
# palette = df[[' r', ' g', ' b']].values
# id2label = classes.to_dict()
# label2id = {v: k for k, v in id2label.items()}

# root_dir = 'drone_dataset'
# feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pathlib import Path


def iter_cv2_images():
    # samples_dir = sys.argv[1]
    samples_dir = "/Users/taichi.muraki/workspace/machine-learning/mur6-lightning-flash-test/data/samples"
    samples_dir = Path(samples_dir)
    sample_images = list(samples_dir.glob("*.jpeg"))
    for p in sample_images:
        image = cv2.imread(str(p))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image


def main():
    model = SegformerForSemanticSegmentation.from_pretrained("./models")
    # model = model.to(device)
    # "deep-learning-analytics/segformer_semantic_segmentation", ignore_mismatched_sizes=True,
    #                                                          num_labels=len(id2label), id2label=id2label, label2id=label2id,
    #                                                          reshape_last_stage=True
    model.eval()
    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    images = []
    for img in iter_cv2_images():
        pixel_values = feature_extractor_inference(img, return_tensors="pt").pixel_values[0].numpy()
        images.append(pixel_values)
    pixel_values = torch.tensor(np.array(images))
    print(pixel_values.shape)
    outputs = model(pixel_values=pixel_values)
    print(outputs)
    logits = outputs.logits.cpu()
    print(logits.shape)
    # First, rescale logits to original image size
    palette = [(0, 0, 0), (128, 64, 128), (130, 76, 0)]

    upsampled_logits = nn.functional.interpolate(
        logits, size=image.shape[:-1], mode="bilinear", align_corners=False  # (height, width)
    )

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3\
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(img)
    axs[1].imshow(color_seg)
    plt.show()


main()
