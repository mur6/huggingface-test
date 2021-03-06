import sys

import matplotlib.pyplot as plt
import torch

from src.utils import get_images

softmax = torch.nn.Softmax()


def main(pt_file, output_image_file):
    logits = torch.load(output_pt_file)
    converted = (softmax(logits) > 0.95).type(torch.uint8)
    # converted = softmax(logits)
    images = get_images()
    fig, axes = plt.subplots(len(images), 3, figsize=(9, 16))
    num = 1
    for ax, orig_image, result_image in zip(axes, images, converted):
        # permute to match the desired memory format
        # result_image = result_image.permute(1, 2, 0).detach().numpy()
        result_image = result_image.detach().numpy()
        save(ax, orig_image, result_image)

    plt.tight_layout()
    plt.savefig(output_image_file, bbox_inches="tight")


def save(ax, orig_image, result_image):
    # fig = plt.figure(figsize=(12, 4))

    # ax[0] = fig.add_subplot(131)
    ax[0].set_title("Original image")
    ax[0].imshow(orig_image)
    # ax2 = fig.add_subplot(132)
    ax[1].set_title("class:1 hand")
    ax[1].imshow(result_image[1], interpolation="none")
    # ax2 = fig.add_subplot(133)
    ax[2].set_title("class:2 mat")
    ax[2].imshow(result_image[2], interpolation="none")

    # plt.savefig(f"output{idx}.png")


if __name__ == "__main__":
    output_pt_file = sys.argv[1]
    output_image_file = sys.argv[2]
    main(output_pt_file, output_image_file)
