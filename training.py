""" Training model for sartorius competition"""

import pandas as pd
import numpy as np
import cv2
import pickle
import pathlib
from utils.misc import get_items_on_path
from utils.vision.transformation import get_image_and_reshape, transform_image_contrast, grayscale_mask
from utils.vision.models import unet_model
import hydra
from omegaconf import OmegaConf

from typing import Tuple, List, TypedDict, Optional
from sklearn.preprocessing import Binarizer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


PATH_PARENT = pathlib.Path(__file__).absolute().parents[0]

class y_holder(TypedDict):
    """Holds raw RLE masks and binary masks"""
    rle_raw: str
    y_transformed: np.ndarray

class X_holder(TypedDict):
    """Holds raw and transformed images """
    image_raw: np.ndarray
    X_transformed: np.ndarray

class X_and_y(TypedDict):
    """Struct for holding X and y unprocessed and preprocessed data"""
    X: X_holder
    y: y_holder


# create typeddict for return value
def data_pipeline(cfg: OmegaConf, images_path: List[pathlib.Path],
                  images_shape: Tuple[int, int], metadata: Optional[pd.DataFrame] = None, 
                  prepare_y: bool =True) -> X_and_y:
    """
    Data pipelines to structure every image and its mask, keeping the raw data and the transformed data.
    Raw data used for exploration and hence enabling feature engineering that goes eventually into the transformed data.
    Transformed are the invidiual X and y structures, ready to be used in a trainings/ validation set.
    """
    
    images_and_ids: List[np.ndarray, str] = [get_image_and_reshape(image_path, images_shape) 
                                            for image_path in images_path]
    
    # create structure of data storage
    data_dict: X_and_y = {
        image_id: {
            "X": {"image_raw": image, "X_transformed": None}, 
            "y": {"rle_raw": None, "y_transformed": None}
        } for image, image_id in images_and_ids}
    
    for image_id in data_dict:
        
        # transform X
        data_dict[image_id]["X"]["X_transformed"] = transform_X(cfg=cfg, image=data_dict[image_id]["X"]["image_raw"])
        
        if prepare_y:
            # get and save rle_encoding for image
            if not metadata:
                raise Exception("metadata for rle have not been passed to the function")
            df_row=metadata[metadata["id"] == image_id]
            annots=df_row["annotation"].tolist()
            data_dict[image_id]["y"]["rle_raw"] = annots
            
            # transform y
            data_dict[image_id]["y"]["y_transformed"] = transform_y(cfg=cfg, rle_mask=data_dict[image_id]["y"]["rle_raw"])
            
    return data_dict


def transform_X(cfg: OmegaConf, image: np.ndarray) -> np.ndarray:
    """Transforms the image and reshaping into 3D structure."""
    
    prepared_x_2d = cv2.resize(
        transform_image_contrast(image, cfg.preprocessing.TRANS_POWER), 
        (cfg.preprocessing.OUTPUT_SHAPE.WIDTH, cfg.preprocessing.OUTPUT_SHAPE.HEIGHT)
        )
    prepared_x_3d = prepared_x_2d.reshape(
        cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1
        )
    return prepared_x_3d


def transform_y(cfg: OmegaConf, rle_mask: List[str]) -> np.ndarray:
    """Transforms the label, ready to be used in training"""
    
    prepared_y_2d = cv2.resize(
        grayscale_mask(rle_mask, (cfg.preprocessing.INPUT_SHAPE.HEIGHT, cfg.preprocessing.INPUT_SHAPE.WIDTH, 1)),
            (cfg.preprocessing.OUTPUT_SHAPE.WIDTH, cfg.preprocessing.OUTPUT_SHAPE.HEIGHT)
        )
    prepared_y_3d = prepared_y_2d.reshape(
        cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1
        )
    prepared_y_3d_binary = Binarizer().transform(prepared_y_3d.reshape(-1, 1)).reshape(prepared_y_3d.shape)

    return prepared_y_3d_binary
        

def prepare_X_and_y(data: X_and_y) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten the data dictionary of X and y"""
    
    X = [data[image_id]["X"]["X_transformed"] for image_id in data]
    y = [data[image_id]["y"]["y_transformed"] for image_id in data]
    
    X = np.array(X)
    y = np.array(y)
    
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
def main(cfg: OmegaConf, preprocess_data_and_cache: bool = False):
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
        train_data = data_pipeline(cfg=cfg, images_path=train_images_paths, 
                                   images_shape= INPUT_IMG_SHAPE, metadata=train_metadata)
        with open(PATH_CACHE_DATA, 'wb') as f:
            pickle.dump(train_data, f, protocol=5)
    else:
        with open(PATH_CACHE_DATA, 'rb') as f:
            train_data = pickle.load(f)
        
    X, y = prepare_X_and_y(train_data)

    model=unet_model(input_img_shape=(cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1))
    model.compile(optimizer = Adam(cfg.training.model.LEARNING_RATE), loss = 'binary_crossentropy', metrics = ['accuracy'])
    train(cfg, model, X, y)
    
    

if __name__ == "__main__":
    main()