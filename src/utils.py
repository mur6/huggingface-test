from pathlib import Path

from PIL import Image


def _iter_pil_images():
    # samples_dir = sys.argv[1]
    samples_dir = "/Users/taichi.muraki/workspace/machine-learning/mur6-lightning-flash-test/data/samples"
    samples_dir = Path(samples_dir)
    sample_images = sorted(list(samples_dir.glob("*.jpeg")))
    for p in sample_images:
        image = Image.open(p)
        yield image


def get_images():
    return list(_iter_pil_images())
