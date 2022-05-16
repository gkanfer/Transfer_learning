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
val_dir = os.path.join(path_origen, 'val_dir')
test_dir = os.path.join(path_origen, 'test_dir')

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

# import pickle

# def save_to_pickle(file_name,var):
#     with open(file_name, 'wb') as handle:
#         pickle.dump(var, handle,protocol=4)
        
# def load_pickle(file_name,var):
#     with open(file_name, 'wb') as handle:
#         pickle.load(var, handle,protocol=4)
    
# save_to_pickle(file_name='train_imgs_scaled.pickle',var=train_imgs_scaled)
# save_to_pickle(file_name='validation_imgs_scaled.pickle',var=validation_imgs_scaled)
# save_to_pickle(file_name='train_labels_enc.pickle',var=validation_labels_enc)
# save_to_pickle(file_name='validation_labels_enc.pickle',var=validation_labels_enc)
# save_to_pickle(file_name='train_imgs.pickle',var=train_imgs)
# save_to_pickle(file_name='validation_imgs.pickle',var=validation_imgs)


'''
tarin labels:['norm', 'pheno', 'norm', 'norm', 'norm'], train_labels_enc:[0 1 0 0 0]
'''
f1 = model_build.model_cnn_basic()
f2 = model_build.model_cnn_basic_Augmentation()
f3 = model_build.model_cnn_transfer_learning_Augmentation_drop_layer_4and5()
f4 = model_build.model_cnn_transfer_learning_Augmentation_freez_all()



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



model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                 input_shape=IMG_DIM))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])

model.summary()

history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch,
                    epochs=epoch,
                    verbose=1)

# save the model
os.chdir(self.path_model)
model.save('cnn_basic.h5')

# test CNN preformance
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1, epoch + 1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epoch + 1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epoch + 1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")




