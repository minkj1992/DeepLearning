import os
import h5py
import image
import operator

import matplotlib.pyplot as plt
import time, pickle, pandas

import numpy as np

import keras
import glob
import PIL
from PIL import Image
from IPython.display import display



from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D ,Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers
from keras import applications

class_name = {
    2: 'Top',
    1: 'Outer',
    0: 'Bottom'}

img_width, img_height = 150, 150
image_input_dir = 'input/'
test_datagen = ImageDataGenerator(rescale=1./255)
inpdata_generator = test_datagen.flow_from_directory(
        image_input_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle = False)
