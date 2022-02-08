""" Training model for sartorius competition"""

import pandas as pd
import numpy as np
import cv2
import pathlib
from utils.misc import get_items_on_path
from utils.vision.transformation import get_image_and_reshape, transform_image_contrast, grayscale_mask
from utils.vision.models import unet_model
import hydra
from omegaconf import OmegaConf

from typing import Tuple, Dict, List
from sklearn.preprocessing import Binarizer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


PATH_PARENT = pathlib.Path(__file__).absolute().parents[0]

# create typeddict for return value
def data_pipeline(cfg: OmegaConf, images_path: List[pathlib.Path], images_shape: Tuple[int, int], metadata: pd.DataFrame) -> dict:
    """ xxxxxxx"""
    
    images_and_ids: List[np.ndarray, str] = [get_image_and_reshape(image_path, images_shape) for image_path in images_path]
    
    # create structure of data storage
    # TODO: use typeddict here
    data_dict = {
        image_id: {
            "X": {"image_raw": image, "X_transformed": None}, 
            "y": {"rle_raw": None, "y_transformed": None}
        } for image, image_id in images_and_ids}
    
    for image_id in data_dict:
        # get and save rle_encoding for image
        df_row=metadata[metadata["id"] == image_id]
        annots=df_row["annotation"].tolist()
        data_dict[image_id]["y"]["rle_raw"] = annots
        
        # transform X
        data_dict[image_id]["X"]["X_transformed"] = preprocess_X(cfg=cfg, image=data_dict[image_id]["X"]["image_raw"])
        
        # transform y
        data_dict[image_id]["y"]["y_transformed"] = preprocess_y(cfg=cfg, rle_mask=data_dict[image_id]["y"]["rle_raw"])
        
    return data_dict


def preprocess_X(cfg: OmegaConf, image: np.ndarray) -> np.ndarray:
    """xxxx"""
    
    prepared_x_2d = cv2.resize(
        transform_image_contrast(image, cfg.preprocessing.TRANS_POWER), 
        (cfg.preprocessing.OUTPUT_SHAPE.WIDTH, cfg.preprocessing.OUTPUT_SHAPE.HEIGHT)
        )
    prepared_x_3d = prepared_x_2d.reshape(
        cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1
        )
    return prepared_x_3d


def preprocess_y(cfg: OmegaConf, rle_mask: List[str]) -> np.ndarray:
    """yyyy"""
    
    prepared_y_2d = cv2.resize(
        grayscale_mask(rle_mask, (cfg.preprocessing.INPUT_SHAPE.HEIGHT, cfg.preprocessing.INPUT_SHAPE.WIDTH, 1)),
            (cfg.preprocessing.OUTPUT_SHAPE.WIDTH, cfg.preprocessing.OUTPUT_SHAPE.HEIGHT)
        )
    prepared_y_3d = prepared_y_2d.reshape(
        cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1
        )
    # TODO: binarize the data
    return prepared_y_3d
        

def prepare_X_and_y(cfg: OmegaConf, ids_and_images: Dict[str, str], metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """ Prepare X and y"""

    image_ids = list(ids_and_images.keys())[:5]
    np.random.shuffle(image_ids)

    X = []
    y = []

    for imd_id in image_ids:

        # prepare X
        image = ids_and_images[imd_id]
        prepared_x_2d = cv2.resize(
            transform_image_contrast(image, cfg.preprocessing.TRANS_POWER), 
            (cfg.preprocessing.OUTPUT_SHAPE.WIDTH, cfg.preprocessing.OUTPUT_SHAPE.HEIGHT)
        )
        prepared_x_3d = prepared_x_2d.reshape(
            cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1
        )
        X.append(prepared_x_3d)

        # prepare y
        annots=metadata[metadata["id"] == imd_id]["annotation"].tolist()
        prepared_y_2d = cv2.resize(
            grayscale_mask(
                annots,
                (cfg.preprocessing.INPUT_SHAPE.HEIGHT, cfg.preprocessing.INPUT_SHAPE.WIDTH, 1)
            ),
            (cfg.preprocessing.OUTPUT_SHAPE.WIDTH, cfg.preprocessing.OUTPUT_SHAPE.HEIGHT)
        )
        prepared_y_3d = prepared_y_2d.reshape(
            cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1
        )
        y.append(prepared_y_3d)

    X = np.array(X)
    y = np.array(y)
    y = Binarizer().transform(y.reshape(-1, 1)).reshape(y.shape) # make y (segmentation labels) binary
    return X, y


def train(cfg: OmegaConf, model: tf.keras.models.Model, X: np.ndarray, y: np.ndarray) -> None:
    """ Main trainings loop execution with callback to tensorboard for logging"""
    
    tf.debugging.set_log_device_placement(True)
    
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=cfg.project_setup.tensorboard.LOG_DIR,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq='epoch',
        profile_batch=(10,20),
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    early_stop=EarlyStopping(
        monitor='val_loss',
        patience=cfg.training.model.PATIENCE,
        restore_best_weights=True)
    
    with tf.device(f'/device:GPU:{cfg.training.device.GPU}'):
        model.fit(
            X, y,
            batch_size = cfg.training.model.BATCH_SIZE,
            epochs = cfg.training.model.EPOCHS,
            validation_split = cfg.training.model.VAL_SIZE,
            callbacks = [early_stop, tb_callback]
        )
        model.save("model")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: OmegaConf, preprocess_data_and_cache: bool = True):
    """ Compose flow and execute"""
    
    print(cfg) # make this a logger
    np.random.seed(cfg.project_setup.RANDOM_STATE_N)
    PATH_TRAIN_IMAGE_FOLDER = PATH_PARENT.joinpath(cfg.project_setup.paths.data.TRAIN_IMAGE_FOLDER)
    PATH_TRAIN_METADATA = PATH_PARENT.joinpath(cfg.project_setup.paths.data.TRAIN_METADATA)
    INPUT_IMG_SHAPE = (cfg.preprocessing.INPUT_SHAPE.HEIGHT, cfg.preprocessing.INPUT_SHAPE.WIDTH)
    PATH_CACHE_DATA = PATH_PARENT.joinpath(cfg.project_setup.paths.data.PREPROCESSED_CACHE)

    # load all train images into memory as dictionary with image_id as key and the image (np.ndarray) as value
    if preprocess_data_and_cache:
        train_metadata = pd.read_csv(PATH_TRAIN_METADATA)
        train_images_paths = get_items_on_path(PATH_TRAIN_IMAGE_FOLDER)
        train_data = data_pipeline(cfg=cfg, images_path=train_images_paths, images_shape= INPUT_IMG_SHAPE, metadata=train_metadata)
        # TODO: save the train_data
        
        # load the train_data again
        # TODO: prepare x and y from the train_data
        X, y = prepare_X_and_y(cfg=cfg, ids_and_images=train_images_dict, metadata=train_metadata)
        

    # loaded = np.load(PATH_CACHE_X_y)
    model=unet_model(input_img_shape=(cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1))
    model.compile(optimizer = Adam(cfg.training.model.LEARNING_RATE), loss = 'binary_crossentropy', metrics = ['accuracy'])
    train(cfg, model, X, y)
    
    

if __name__ == "__main__":
    main()