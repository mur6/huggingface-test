import os
import pandas as pd
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
import math

# df = pd.read_csv('drone_dataset/class_dict_seg.csv')
# classes = df['name']
# palette = df[[' r', ' g', ' b']].values
# id2label = classes.to_dict()
# label2id = {v: k for k, v in id2label.items()}

# root_dir = 'drone_dataset'
# feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SegformerForSemanticSegmentation.from_pretrained("deep-learning-analytics/segformer_semantic_segmentation", ignore_mismatched_sizes=True,
#                                                          num_labels=len(id2label), id2label=id2label, label2id=label2id,
#                                                          reshape_last_stage=True)
# model = model.to(device)
