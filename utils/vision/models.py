from typing import Tuple
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, SpatialDropout2D, Concatenate, LeakyReLU
from tensorflow.keras.models import Model


def unet_model(input_img_shape: Tuple[int,int,int]):
    """ Keras UNET model"""

    input_layer = Input(shape = input_img_shape, name = 'Input_Layer')
    
    conv_1 = Conv2D(16, 5, padding = 'same', activation = LeakyReLU(), name = 'Conv_1')(input_layer)
    pool_1 = MaxPool2D(name = 'Max_Pool_1')(conv_1)
    spd_1 = SpatialDropout2D(0.1, name = 'SPD_1')(pool_1)
    
    conv_2 = Conv2D(32, 4, padding = 'same', activation = LeakyReLU(), name = 'Conv_2')(spd_1)
    pool_2 = MaxPool2D(name = 'Max_Pool_2')(conv_2)  
    conv_3 = Conv2D(64, 4, padding = 'same', activation = LeakyReLU(), name = 'Conv_3')(pool_2)
    pool_3 = MaxPool2D(name = 'Max_Pool_3')(conv_3)
    spd_2 = SpatialDropout2D(0.1, name = 'SPD_2')(pool_3)
    
    conv_4 = Conv2D(128, 3, padding = 'same', activation = LeakyReLU(), name = 'Conv_4')(spd_2)
    pool_4 = MaxPool2D(name = 'Max_Pool_4')(conv_4)
    conv_5 = Conv2D(256, 3, padding = 'same', activation = LeakyReLU(), name = 'Conv_5')(pool_4)
    pool_5 = MaxPool2D(name = 'Max_Pool_5')(conv_5)
    spd_3 = SpatialDropout2D(0.1, name = 'SPD_3')(pool_5)
    
    conv_6 = Conv2D(512, 2, padding = 'same', activation = LeakyReLU(), name = 'Conv_6')(spd_3)
    pool_6 = MaxPool2D(name = 'Max_Pool_6')(conv_6)
    
    conv_t_1 = Conv2DTranspose(256, 1, padding = 'same', strides = 2, activation = LeakyReLU(), name = 'Conv_T_1')(pool_6)
    concat_1 = Concatenate(name = 'Concat_1')([conv_t_1, spd_3])
    spd_4 = SpatialDropout2D(0.1, name = 'SPD_4')(concat_1)
    
    conv_t_2 = Conv2DTranspose(128, 3, padding = 'same', strides = 2, activation = LeakyReLU(), name = 'Conv_T_2')(spd_4)
    conv_t_3 = Conv2DTranspose(64, 3, padding = 'same', strides = 2, activation = LeakyReLU(), name = 'Conv_T_3')(conv_t_2)
    concat_2 = Concatenate(name = 'Concat_2')([conv_t_3, spd_2])
    spd_5 = SpatialDropout2D(0.1, name = 'SPD_5')(concat_2)
    
    conv_t_4 = Conv2DTranspose(32, 4, padding = 'same', strides = 2, activation = LeakyReLU(), name = 'Conv_T_4')(spd_5)
    conv_t_5 = Conv2DTranspose(16, 4, padding = 'same', strides = 2, activation = LeakyReLU(), name = 'Conv_T_5')(conv_t_4)
    concat_3 = Concatenate(name = 'Concat_3')([conv_t_5, spd_1])
    spd_6 = SpatialDropout2D(0.1, name = 'SPD_6')(concat_3)
    
    conv_t_6 = Conv2DTranspose(8, 5, padding = 'same', strides = 2, activation = LeakyReLU(), name = 'Conv_T_6')(spd_6)
    
    output_layer = Conv2DTranspose(1, 5, padding = 'same', activation = 'sigmoid', name = 'Output_Layer')(conv_t_6)
    
    return Model(inputs = input_layer, outputs = output_layer, name = 'Sartorius')
