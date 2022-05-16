#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:01:26 2022

@author: kanferg
"""
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

import glob
import numpy as np
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

from keras.applications import vgg16
from keras.models import Model
import keras
import pandas as pd


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from tensorflow.keras import optimizers

import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
import os

batch_size  = 30
epoch  = 100
#step_per_epoch = train length / batch size
step_per_epoch = int((9217*2)/30)
validation_steps = int(1153*2)/30
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models'


path_input = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/input_sc_mix'
path_origen = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set'

train_dir = os.path.join(path_origen, 'training_data')
val_dir = os.path.join(path_origen, 'val_dir')
test_dir = os.path.join(path_origen, 'test_dir')
IMG_DIM = (150,150,3)





train_files = glob.glob(os.path.join(train_dir,'*.' + 'png'))
train_imgs = [img_to_array(load_img(img, target_size=(150,150,3))) for img in train_files]

train_imgs = []
x=0
for img in train_files:
    while x<3:
        x +=1
        try:
            train_imgs.append(img_to_array(load_img(img, target_size=(150,150,3))))
        except:
            continue
           

         


train_imgs = np.array(train_imgs)
train_imgs = train_imgs
train_labels = [fn.split('/')[1].split('_')[0].strip() for fn in train_files]
# Validation file generator
validation_files = glob.glob(os.path.join(path_validation,'*.' + file_extention))
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
self.validation_imgs = validation_imgs
validation_labels = [fn.split('/')[1].split('_')[0].strip() for fn in validation_files]