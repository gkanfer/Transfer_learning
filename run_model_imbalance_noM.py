#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run on no machine
"""
import shutil
import numpy as np
#import glob
#from sklearn.model_selection import train_test_split
import os
os.chdir('/data/kanferg/Images/Pex_project/Transfer_learning/code')
from utils import Taining_data_orgenizer as orgenizer 
from utils import model_builder as mb
import shutil
#import glob
#from sklearn.model_selection import train_test_split
import os
os.chdir('/data/kanferg/Images/Pex_project/Transfer_learning/code')
from utils import Taining_data_orgenizer as orgenizer 
from utils import model_builder as mb

path_input = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/input_sc_mix'
path_origen = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set'

train_dir = os.path.join(path_origen, 'training_data')
val_dir = os.path.join(path_origen, 'validation_data')
test_dir = os.path.join(path_origen, 'test_data')

# 10%

path_input = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/input_sc_mix'
path_origen = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set'

train_dir = os.path.join(path_origen, 'training_data')
val_dir = os.path.join(path_origen, 'validation_data')
test_dir = os.path.join(path_origen, 'test_data')

'''
batch size, epochs, steps_per_epoch_sel, validation_steps 
'''
batch  = 30
epoch  = 100
step_per_epoch = int((9930)/30)
validation_steps = int((1242)/30)
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models/10precent'
# extract_size_train = 1000
# extract_size_val = 200
IMG_DIM=(150,150,3)
imbalance_train = 921
imbalance_val = 115

'''
load image files (1000 for training and 200  for validation)
'''

model_build = mb.model_builder(IMG_DIM=(150,150,3),path_training=train_dir,path_validation=val_dir,
                 batch=batch, epoch = epoch,input_shape = (150,150,3) ,steps_per_epoch_sel= step_per_epoch,
                 validation_steps=validation_steps,path_model = path_model,file_extention = 'png',
                 imbalance_train = imbalance_train,
                              imbalance_val = imbalance_val)

train_imgs_scaled, validation_imgs_scaled,train_labels_enc,validation_labels_enc,train_imgs,validation_imgs,report = model_build.build_image__sets()
model_imbalance = model_build.model_cnn_transfer_learning_Augmentation_drop_layer_4and5()



'''
batch size, epochs, steps_per_epoch_sel, validation_steps 
'''
batch  = 30
epoch  = 100
step_per_epoch = int((8000)/30)
validation_steps = int((2000)/30)
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models/20precent'
IMG_DIM=(150,150,3)
imbalance_train = 2000
imbalance_val = 300

'''
load image files (1000 for training and 200  for validation)
'''

model_build = mb.model_builder(IMG_DIM=(150,150,3),path_training=train_dir,path_validation=val_dir,
                 batch=batch, epoch = epoch,input_shape = (150,150,3) ,steps_per_epoch_sel= step_per_epoch,
                 validation_steps=validation_steps,path_model = path_model,file_extention = 'png',
                 imbalance_train = imbalance_train,
                              imbalance_val = imbalance_val)
train_imgs_scaled, validation_imgs_scaled,train_labels_enc,validation_labels_enc,train_imgs,validation_imgs,report = model_build.build_image__sets()
#print('number of training samples pheno: {}'.format(np.sum(train_labels_enc)))
#print('number of training samples norm: {}'.format(np.abs(np.sum(train_labels_enc-1))))
model_imbalance_20percent = model_build.model_cnn_transfer_learning_Augmentation_drop_layer_4and5()
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models'
os.chdir(path_model)
