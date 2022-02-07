import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from utils.vision.transformation import grayscale_mask, rgb_mask

def plot_image_with_annot(image: np.ndarray, rle_masks: str, img_2d_shape: Tuple[int, int], normalize_hist:bool=True) -> None:
    """Plot image with rle-mask in greyscale and in rbg"""

    grayscale_masks=grayscale_mask(rle_masks, shape=(*img_2d_shape, 1))
    rgb_masks=rgb_mask(rle_masks, shape=(*img_2d_shape, 3))

    plt.figure(figsize = (20 , 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap = "gray")
    plt.title("Original Image", fontsize = 16)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(image * grayscale_masks, cmap = "gray")
    plt.title('Input image with mask', fontsize = 16)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(rgb_masks)
    plt.title('RGB mask', fontsize = 16)
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    if normalize_hist:
        plt.hist(image.flatten() / 255, bins = 255, range = (0, 1))
    else:
        plt.hist(image.flatten(), bins=255, range=(0, 1))
    plt.title('Pixel distribution', fontsize = 16)
    
    plt.suptitle("For a sample image, masks and their pixel distributions", fontsize = 20)
    plt.tight_layout(rect = [0, 0, 0.90, 1])
    plt.show()

