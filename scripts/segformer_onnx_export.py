# import cv2
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin

from src.utils import get_images


class WrapperModel(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
        self.softmax = torch.nn.Softmax()

    def forward(self, pixel_values_pt):
        return_tensors = "pt"
        encoded_inputs = BatchFeature(data={"pixel_values": pixel_values_pt}, tensor_type=return_tensors)
        print(type(encoded_inputs), type(encoded_inputs["pixel_values"]))
        result = self.model(**encoded_inputs)
        logits = result.logits
        # print(type(a))
        converted = (self.softmax(logits) > 0.95).type(torch.uint8)
        converted.squeeze_(0)
        print(converted[1].shape, converted[2].shape)
        return {"hand": converted[1], "mat": converted[2]}


def main():
    images = get_images()[0]

    # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    batch_feature = feature_extractor_inference(images=images, return_tensors="pt")
    print(type(batch_feature), type(batch_feature["pixel_values"]))
    model_dir = "models/mit-b1/"
    model = WrapperModel(model_dir)
    model.eval()
    # print(type(input_val))
    # outputs = model(batch_feature["pixel_values"]) #model(**inputs)
    # # model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
    # # inputs = feature_extractor_inference(images=images, return_tensors="pt")
    torch.onnx.export(
        model,
        batch_feature["pixel_values"],
        "models/onnx/segformer.onnx",
        input_names=["input"],
        output_names=["hand", "mat"],
    )


main()
