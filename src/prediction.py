import pathlib
from typing import Tuple, List

from tensorflow import keras
import numpy as np
import pandas as pd

from utils.misc import get_items_on_path
from utils.vision.transformation import rle_encode
from src.training import data_pipeline, prepare_X_and_y

from sklearn.preprocessing import Binarizer

from hydra import initialize, compose
initialize("./configs")

cfg = compose(config_name="config.yaml")
np.random.seed(cfg.project_setup.RANDOM_STATE_N)


PATH_PARENT = pathlib.Path("__file__").absolute().parents[0]
PATH_TEST_DATA = PATH_PARENT.joinpath("../input/sartorius-cell-instance-segmentation/test")
PATH_MODEL = PATH_PARENT.joinpath("outputs", "2022-02-11", "05-54-14", "model")
INPUT_IMG_SHAPE = (cfg.preprocessing.INPUT_SHAPE.HEIGHT, cfg.preprocessing.INPUT_SHAPE.WIDTH)

def predict() -> Tuple[List[str], List[str]]:
    """Load model, data and predict"""
    
    # load and transform test data
    test_images_paths = get_items_on_path(PATH_TEST_DATA)
    test_data = data_pipeline(cfg=cfg, images_path=test_images_paths, 
                        images_shape=INPUT_IMG_SHAPE, prepare_y=False)
    X = [test_data[image_id]["X"]["X_transformed"] for image_id in test_data]
    X = np.array(X)

    # load trained model
    model = keras.models.load_model(PATH_MODEL)
    
    # prediction loop
    image_ids = []
    rle_encodings = []
    for image_id in test_data:
        X_transformed = test_data[image_id]["X"]["X_transformed"]
        pred_y = model.predict(np.array([X_transformed,]))
        pred_y_mask = Binarizer(threshold = 0.4).transform(pred_y.reshape(-1, 1)).reshape(pred_y.shape)
        rle_encoding = rle_encode(pred_y_mask)
        
        image_ids.append(image_id)
        rle_encodings.append(rle_encoding)
        

    return image_ids, rle_encodings


if __name__ == "__main__":
    image_ids, rle_encodings = predict()
    
    submission = pd.DataFrame({
        'id': image_ids,
        'predicted': rle_encodings
    })
    
    submission.to_csv("submission.csv", index=False)
    
