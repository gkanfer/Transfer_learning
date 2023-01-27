import shutil
#import glob
#from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as CustomLogisticRegression
import plotnine
from plotnine import *
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.models import load_model

os.chdir('/data/kanferg/Images/Pex_project/Transfer_learning/code')
from utils import Taining_data_orgenizer as orgenizer 
from utils import model_builder as mb


path_input = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/input_sc_mix'
path_origen = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set'

train_dir = os.path.join(path_origen, 'training_data')
val_dir = os.path.join(path_origen, 'validation_data')
test_dir = os.path.join(path_origen, 'test_data')
# basic
batch  = 30
epoch  = 50
step_per_epoch = int((9930)/30)
validation_steps = int((1242)/30)
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models'
# extract_size_train = 1000
# extract_size_val = 200
IMG_DIM=(150,150,3)
imbalance_train = 921
imbalance_val = 115
model_name = '10precent.h5'
path_checkpoints = '/data/kanferg/Images/Pex_project/Transfer_learning/models/chakpoints_10p/'

model_build = mb.model_builder(IMG_DIM=(150,150,3),path_training=train_dir,path_validation=val_dir,
                 batch=batch, epoch = epoch,input_shape = (150,150,3) ,steps_per_epoch_sel= step_per_epoch,
                 validation_steps=validation_steps,path_model = path_model,file_extention = 'png',
                 imbalance_train = imbalance_train, imbalance_val = imbalance_val,model_name=model_name, path_checkpoints=path_checkpoints)
train_imgs_scaled, validation_imgs_scaled,train_labels_enc,validation_labels_enc,train_imgs,validation_imgs,report = model_build.build_image__sets()
model_imbalance = model_build.model_cnn_transfer_learning_Augmentation_drop_layer_4and5()

batch  = 30
epoch  = 50
step_per_epoch = int((8000)/30)
validation_steps = int((2000)/30)
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models'
IMG_DIM=(150,150,3)
imbalance_train = 2000
imbalance_val = 300
model_name = '20precent.h5'
path_checkpoints = '/data/kanferg/Images/Pex_project/Transfer_learning/models/chakpoints_20p/'

model_build = mb.model_builder(IMG_DIM=(150,150,3),path_training=train_dir,path_validation=val_dir,
                 batch=batch, epoch = epoch,input_shape = (150,150,3) ,steps_per_epoch_sel= step_per_epoch,
                 validation_steps=validation_steps,path_model = path_model,file_extention = 'png',
                 imbalance_train = imbalance_train, imbalance_val = imbalance_val,model_name=model_name, path_checkpoints=path_checkpoints)
train_imgs_scaled, validation_imgs_scaled,train_labels_enc,validation_labels_enc,train_imgs,validation_imgs,report = model_build.build_image__sets()
model_imbalance = model_build.model_cnn_transfer_learning_Augmentation_drop_layer_4and5()





