#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate models
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

os.chdir('/data/kanferg/Images/Pex_project/Transfer_learning/code')
from utils import model_evaluation_utils as meu

############ Load test image set #############

IMG_DIM = (150, 150)
os.chdir("/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/")
test_files = glob.glob('test_data/*')
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
#test_files[0].split('/')[1].split('_')[0].strip()
test_labels = [fn.split('/')[1].split('_')[0].strip() for fn in test_files]
#test_labels = [fn.split('/')[1].split('.')[0].strip() for fn in test_files]

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
num2class_label_transformer = lambda l: ['norm' if x[0] == 0 else 'pheno' for x in l]
class2num_label_transformer = lambda l: [0 if x == 'pheno' else 1 for x in l]
test_labels_enc = class2num_label_transformer(test_labels[:5])

print('Test dataset shape:{}'.format(test_imgs.shape))
print(test_labels[0:5], test_labels_enc[0:5])

'''
Test dataset shape:(5967, 150, 150, 3)
['norm', 'pheno', 'norm', 'pheno', 'norm'] [1, 0, 1, 0, 1]

'''

############ Load models #############

path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models'
os.chdir(path_model)

cnn_basic = load_model('cnn_basic.h5')
cnn_basic_Augmentation = load_model('cnn_basic_Augmentation.h5')
cnn_transfer_learning_Augmentation_drop_layer_4and5 = load_model('cnn_transfer_learning_Augmentation_drop_layer_4and5.h5')
transfer_learning_aug_dropout_freez_all = load_model('transfer_learning_aug_dropout_freez_all.h5')

############ predictions #############

predictions = cnn_basic.predict(test_imgs_scaled, verbose=0)
plt.hist(predictions)
classes_x=[np.where(lab >0.5,1,0).tolist() for lab in predictions]
predictions_label = num2class_label_transformer(classes_x)
predictions_label

meu.get_metrics(test_labels, predictions_label)

'''
Accuracy: 0.9878
Precision: 0.9883
Recall: 0.9878
F1 Score: 0.9879
'''

predictions = cnn_basic_Augmentation.predict(test_imgs_scaled, verbose=0)
plt.hist(predictions)
classes_x=[np.where(lab >0.5,1,0).tolist() for lab in predictions]
predictions_label = num2class_label_transformer(classes_x)
predictions_label

meu.get_metrics(test_labels, predictions_label)

'''
Accuracy: 0.9894
Precision: 0.9895
Recall: 0.9894
F1 Score: 0.9895
'''

predictions = cnn_transfer_learning_Augmentation_drop_layer_4and5.predict(test_imgs_scaled, verbose=0)
plt.hist(predictions)
classes_x=[np.where(lab >0.5,1,0).tolist() for lab in predictions]
predictions_label = num2class_label_transformer(classes_x)
predictions_label

meu.get_metrics(test_labels, predictions_label)

'''
Accuracy: 0.9953
Precision: 0.9953
Recall: 0.9953
F1 Score: 0.9953
'''

predictions = transfer_learning_aug_dropout_freez_all.predict(test_imgs_scaled, verbose=0)
plt.hist(predictions)
classes_x=[np.where(lab >0.5,1,0).tolist() for lab in predictions]
predictions_label = num2class_label_transformer(classes_x)
predictions_label

meu.get_metrics(test_labels, predictions_label)

'''
Accuracy: 0.9834
Precision: 0.9841
Recall: 0.9834
F1 Score: 0.9836
'''

import skimage 
os.chdir("/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set")
plt.imshow(skimage.io.imread(test_files[5]))





