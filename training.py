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

from typing import Tuple, Dict
from sklearn.preprocessing import Binarizer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


PATH_PARENT = pathlib.Path(__file__).absolute().parents[0]

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
def main(cfg: OmegaConf, preprocess_data: bool = True):
    """ Compose flow and execute"""
    
    print(cfg) # make this a logger
    np.random.seed(cfg.project_setup.RANDOM_STATE_N)
    PATH_TRAIN_IMAGE_FOLDER = PATH_PARENT.joinpath(cfg.project_setup.paths.data.TRAIN_IMAGE_FOLDER)
    PATH_TRAIN_METADATA = PATH_PARENT.joinpath(cfg.project_setup.paths.data.TRAIN_METADATA)
    INPUT_IMG_SHAPE = (cfg.preprocessing.INPUT_SHAPE.HEIGHT, cfg.preprocessing.INPUT_SHAPE.WIDTH)
    PATH_CACHE_X_y = PATH_PARENT.joinpath(cfg.project_setup.paths.data.PREPROCESSED_CACHE)

    # load all train images into memory as dictionary with image_id as key and the image (np.ndarray) as value
    if preprocess_data:
        train_metadata = pd.read_csv(PATH_TRAIN_METADATA)
        train_images_paths = get_items_on_path(PATH_TRAIN_IMAGE_FOLDER)
        train_images = [get_image_and_reshape(train_image_path, INPUT_IMG_SHAPE) for train_image_path in train_images_paths]
        train_images_dict = {image_id: image for image, image_id in train_images}
        X, y = prepare_X_and_y(cfg=cfg, ids_and_images=train_images_dict, metadata=train_metadata)
        np.savez_compressed(PATH_CACHE_X_y, X=X, y=y)

    loaded = np.load(PATH_CACHE_X_y)
    model=unet_model(input_img_shape=(cfg.preprocessing.OUTPUT_SHAPE.HEIGHT, cfg.preprocessing.OUTPUT_SHAPE.WIDTH, 1))
    model.compile(optimizer = Adam(cfg.training.model.LEARNING_RATE), loss = 'binary_crossentropy', metrics = ['accuracy'])
    train(cfg, model, X, y)
    
    

if __name__ == "__main__":
    main()