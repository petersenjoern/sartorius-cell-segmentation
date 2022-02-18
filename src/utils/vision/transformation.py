import pathlib
from typing import Tuple
import numpy as np
import cv2

def get_image_and_reshape(path_single_image: pathlib, image_shape: Tuple[int, int], image_format:str="png") -> Tuple[np.ndarray, str]:
    """Function to load image and reshape to desired shape"""
    
    image_id = path_single_image.name.replace(f".{image_format}", "")
    image=cv2.imread(str(path_single_image), cv2.IMREAD_GRAYSCALE) #cv2 imread takes only str
    return image.reshape(*image_shape, 1), image_id


def rle_decode(mask_rle: str,  shape: Tuple[int, int, int], color: int = 1) -> np.ndarray:
    """
    Run-Length decoding.

    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return 
    returns numpy array, 1 - mask, 0 - background
    """ 

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros((shape[0] * shape[1], shape[2]), dtype = np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)


def rle_encode(image: np.ndarray) -> str:
    """ 
    Run-length encoding.
    
    image: np.ndarray of shape Tuple[int, int, int] with 1 - mask, 0 - background
    returns run-length as string formatted (start length)
    """
    pixels = image.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    

def grayscale_mask(annots: str, shape: Tuple[int, int, int]) -> np.ndarray:
    """ Create grayscale mask from rle-mask(s). run-length-mask are encoded as string (start length)"""

    grayscale_mask = np.zeros((shape[0], shape[1], shape[2]))
    
    for annot in annots:
            grayscale_mask += rle_decode(annot, shape)
    
    return grayscale_mask.clip(0, 1)


def rgb_mask(annots: str, shape: Tuple[int, int, int]) -> np.ndarray:
    """ Create RGB mask from rle-mask(s). run-length-mask are encoded as string (start length)"""

    rgb_mask = np.zeros((shape[0], shape[1], shape[2]))
    
    for annot in annots:
        rgb_mask += rle_decode(mask_rle=annot, shape=shape, color=np.random.rand(3))
    
    return rgb_mask.clip(0, 1)

def transform_image_contrast(img_data: np.ndarray, power:int = 2) -> np.ndarray:
    img_data_mask = np.ones_like(img_data, dtype = np.int16)
    img_data_mask[img_data < 127.5] = -1
    
    img_data_transformed = img_data.astype(np.int16) - 127.5
    img_data_transformed[img_data_transformed > 0] = np.power(img_data_transformed[img_data_transformed > 0], 1 / power)
    img_data_transformed[img_data_transformed < 0] = np.power(-img_data_transformed[img_data_transformed < 0], 1 / power)
    img_data_transformed = ((img_data_transformed * img_data_mask) / (2 * np.power(127.5, 1 / power))) + 0.5
    
    return img_data_transformed