import pathlib
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def get_items_on_path(path: pathlib.Path) -> List[pathlib.Path]:
    """
    Function to combine directory path with individual items on path
    """

    items_on_path = []
    for filepath in path.iterdir():
        items_on_path.append(filepath)
    return items_on_path


def plot_multiple_img(images: List[np.ndarray], rows: int, cols: int) -> None:
    """
    Display images from dataset.
    """
    
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16,8))
    for ind, image in enumerate(images):        
        ax.ravel()[ind].imshow(image)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()


