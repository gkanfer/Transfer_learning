#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constracting models
"""
import os
os.chdir('/data/kanferg/Images/Pex_project/Transfer_learning/code')
from utils import Taining_data_orgenizer as orgenizer 
from utils import model_builder as mb

path_input = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/input_sc_mix'
path_origen = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set'

train_dir = os.path.join(path_origen, 'training_data')
val_dir = os.path.join(path_origen, 'validation_data')
test_dir = os.path.join(path_origen, 'test_data')

'''
batch size, epochs, steps_per_epoch_sel, validation_steps 
'''
# import tensorflow as tf
# tf.config.experimental.get_memory_info('GPU:0')
#available GPU memory bytes / 4 / (size of tensors + trainable parameters)
batch  = 30
epoch  = 100
#step_per_epoch = train length / batch size
step_per_epoch = int((9217*2)/30)
validation_steps = int((1153*2)/30)
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models'
extract_size_train = 1000
extract_size_val = 200
IMG_DIM=(150,150,3)

'''
load image files (1000 for training and 200  for validation)
'''

model_build = mb.model_builder(IMG_DIM=(150,150,3),path_training=train_dir,path_validation=val_dir,
                 batch=batch, epoch = epoch,input_shape = (150,150,3) ,steps_per_epoch_sel= step_per_epoch,
                 validation_steps=validation_steps,path_model = path_model,file_extention = 'png',
                 extract_size_train = extract_size_train, extract_size_val=extract_size_val)



train_imgs_scaled, validation_imgs_scaled,train_labels_enc,validation_labels_enc,train_imgs,validation_imgs,report = model_build.build_image__sets()



'''
load image files 
'''

model_build = mb.model_builder(IMG_DIM=(150,150,3),path_training=train_dir,path_validation=val_dir,
                 batch=batch, epoch = epoch,input_shape = (150,150,3) ,steps_per_epoch_sel= step_per_epoch,
                 validation_steps=validation_steps,path_model = path_model,file_extention = 'png')



train_imgs_scaled, validation_imgs_scaled,train_labels_enc,validation_labels_enc,train_imgs,validation_imgs,report = model_build.build_image__sets()





'''
tarin labels:['norm', 'pheno', 'norm', 'norm', 'norm'], train_labels_enc:[0 1 0 0 0]
'''
f1 = model_build.model_cnn_basic()
f2 = model_build.model_cnn_basic_Augmentation()
f3 = model_build.model_cnn_transfer_learning_Augmentation_drop_layer_4and5()
f4 = model_build.model_cnn_transfer_learning_Augmentation_freez_all()







