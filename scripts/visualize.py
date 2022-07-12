import matplotlib.pyplot as plt
import torch

from src.utils import get_images

softmax = torch.nn.Softmax()


def main():
    logits = torch.load("logits_mit_b4.pt")
    converted = (softmax(logits) > 0.95).type(torch.uint8)
    # converted = softmax(logits)
    num = 1
    for orig_image, result_image in zip(get_images(), converted):
        # permute to match the desired memory format
        # result_image = result_image.permute(1, 2, 0).detach().numpy()
        result_image = result_image.detach().numpy()
        save(num, orig_image, result_image)
        num += 1


def save(idx, orig_image, result_image):
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131)
    ax1.set_title("Original image")
    ax1.imshow(orig_image)
    ax2 = fig.add_subplot(132)
    ax2.set_title("class:1 hand")
    ax2.imshow(result_image[1], cmap="viridis", interpolation="none")
    ax2 = fig.add_subplot(133)
    ax2.set_title("class:2 mat")
    ax2.imshow(result_image[2], cmap="viridis", interpolation="none")

    plt.savefig(f"output{idx}.png")


main()
