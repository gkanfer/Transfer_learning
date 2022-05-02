#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transfer-learning-with-keras
Chapter 4 from applied deep learning book
/Desktop/NIH_Youle/Lab%20book/Advanced_applied_deep_learning_book/chapter4/Transfer-learning-with-keras.ipynb
https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
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

np.random.seed(42)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

"""
creat a training and validation folders
1) add the folder name to the image name
2) count image number 
"""
path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/"
files = glob.glob(path_orig+"*/*.png")
"""
len(files)
11137
"""
#adding prefix acording to the directory prefix
# import os

# for root, dirs, files in os.walk(path_orig):
#     if not files:
#         continue
#     prefix = os.path.basename(root)
#     for f in files:
#         os.rename(os.path.join(root, f), os.path.join(root, "{}_{}".format(prefix, f)))

# # count Cytosole vs Nuclues
# files = glob.glob(path_orig+"*/*.png")
# nuc_files = [fn for fn in files if 'Cytosole' in fn]
# cyto_files = [fn for fn in files if 'Nuclues' in fn]
# len(nuc_files), len(cyto_files)
# #(4967, 6170)

# # We build a smaller dataset
# nuc_train = np.random.choice(nuc_files, size=3000, replace=False)
# cyto_train = np.random.choice(cyto_files, size=3000, replace=False)
# nuc_files = list(set(nuc_files) - set(nuc_train))
# cyto_files = list(set(cyto_files) - set(cyto_train))

# nuc_val = np.random.choice(nuc_files, size=800, replace=False)
# cyto_val = np.random.choice(cyto_files, size=800, replace=False)
# nuc_files = list(set(nuc_files) - set(nuc_val))
# cyto_files = list(set(cyto_files) - set(cyto_val))

# nuc_test = np.random.choice(nuc_files, size=800, replace=False)
# cyto_test = np.random.choice(cyto_files, size=800, replace=False)

# print('nuc datasets:', nuc_train.shape, nuc_val.shape, nuc_test.shape)
# print('cyto datasets:', cyto_train.shape, cyto_val.shape, cyto_test.shape)
'''
nuc datasets: (3000,) (800,) (800,)
cyto datasets: (3000,) (800,) (800,)
'''
os.chdir(path_orig)
train_dir = 'training_data'
val_dir = 'validation_data'
test_dir = 'test_data'

# train_files = np.concatenate([nuc_train, cyto_train])
# validate_files = np.concatenate([nuc_val, cyto_val])
# test_files = np.concatenate([nuc_test, cyto_test])

# os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
# os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
# os.mkdir(test_dir) if not os.path.isdir(test_dir) else None

# for fn in train_files:
#     shutil.copy(fn, train_dir)

# for fn in validate_files:
#     shutil.copy(fn, val_dir)
    
# for fn in test_files:
#     shutil.copy(fn, test_dir)

'''
Load of the actual images
'''
IMG_DIM = (150, 150)

train_files = glob.glob('training_data/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('/')[1].split('_')[0].strip() for fn in train_files]
# MAC '/', WINDOWS '\\'

validation_files = glob.glob('validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('/')[1].split('_')[0].strip() for fn in validation_files]

print('Train dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)
#Train dataset shape: (6000, 150, 150, 3) 	Validation dataset shape: (1600, 150, 150, 3)
train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape) #(150, 150, 3)
array_to_img(train_imgs[0]) #dispaly the first image
'''
change labale to 0 for nuc and 1 to cyto
'''
batch_size = 30
num_classes = 2

#Training data of 6000 with epoch of 30 
#200 iteration per epoch

epochs = 30
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[10:15], train_labels_enc[10:15])

'''
CNN model
'''
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                 input_shape=input_shape))
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
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)


#test CNN preformance
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

#clearly our model is over fitting
'''
CNN Model with Image Augmentation
'''
# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
#                                    horizontal_flip=True, fill_mode='wrap',brightness_range=[0.1,0.9])

# val_datagen = ImageDataGenerator(rescale=1./255)

# # show how  images looking after agumantation was applied
# img_id = 2000
# tfeb_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],
#                                    batch_size=1)
# temp = [next(tfeb_generator) for i in range(0,9)]
# fig, ax = plt.subplots(1,9, figsize=(16, 6))
# print('Labels:', [item[1][0] for item in temp])
# l = [ax[i].imshow(temp[i][0][0]) for i in range(0,9)]

# #test just the brightness
# train_datagen = ImageDataGenerator(rescale=1./255,fill_mode='wrap', brightness_range=[0.1,0.9])
# val_datagen = ImageDataGenerator(rescale=1./255)

# img_id = 2000
# tfeb_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],
#                                    batch_size=1)
# temp = [next(tfeb_generator) for i in range(0,9)]
# fig, ax = plt.subplots(1,9, figsize=(16, 6))
# print('Labels:', [item[1][0] for item in temp])
# l = [ax[i].imshow(temp[i][0][0]) for i in range(0,9)]

# here is the agumantation I chosse to do:
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='wrap',brightness_range=[0.1,0.9])

val_datagen = ImageDataGenerator(rescale=1./255)

# applay new model with droput and aougmentation
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
input_shape = (150, 150, 3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])
              
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50, 
                              verbose=1)  

#save the model
os.chdir("/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp")
model.save('tfeb_aug_dropout.h5')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epochs=100
epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

'''
Transfer learning beased on Vgg16 (5 conv blocks)
The weigts used are from imagenet
1) we will first freez all blocks and test the preformance
2) Then, we will use just feture 4/5
'''


from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
    
import pandas as pd
#show which layer are frozen
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']) 
#Image example fead to block number 5
bottleneck_feature_example = vgg.predict(train_imgs_scaled[0:1])
print(bottleneck_feature_example.shape)
plt.imshow(bottleneck_feature_example[0][:,:,0])

#flat all the fetures from training and validation set 
#for feeding the last blovk of vg16
def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features
    
train_features_vgg = get_bottleneck_features(vgg_model, train_imgs_scaled)
validation_features_vgg = get_bottleneck_features(vgg_model, validation_imgs_scaled)

print('Train Bottleneck Features:', train_features_vgg.shape, 
      '\tValidation Bottleneck Features:', validation_features_vgg.shape)
#results:
    #Train Bottleneck Features: (6000, 8192) 	
    #Validation Bottleneck Features: (1600, 8192)
'''
now we will train with aougmentation
Pre-trained CNN model as a Feature Extractor with Image Augmentation
'''
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='wrap',brightness_range=[0.1,0.9])

val_datagen = ImageDataGenerator(rescale=1./255)

# applay new model with droput and aougmentation
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
input_shape = (150, 150, 3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
              
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50, 
                              verbose=1)  

#save the model
os.chdir("/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp")
model.save('tfeb_aug_dropout_vgg16_freezall.h5')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('tfeb_aug_dropout_vgg16_freezall', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epochs=100
epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

'''
Train on Vgg16 with unfreez con block4/5
'''
vgg_model.trainable = True

set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']) 

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='wrap',brightness_range=[0.1,0.9])

val_datagen = ImageDataGenerator(rescale=1./255)

# applay new model with droput and aougmentation
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
input_shape = (150, 150, 3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
              
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50, 
                              verbose=1)  

os.chdir("/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp")
model.save('tfeb_aug_dropout_vgg16_unfreezblock_4and5.h5')

'''
Evaluating our Deep Learning Models on Test Data
'''

# load dependencies
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
import os
os.chdir("/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/code")
import utils.model_evaluation_utils as meu
%matplotlib inline

# load saved models
os.chdir("/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp")
basic_cnn = load_model('tfeb_aug_dropout.h5')
#img_aug_cnn = load_model('cats_dogs_cnn_img_aug.h5')
#tl_cnn = load_model('cats_dogs_tlearn_basic_cnn.h5')
tl_img_aug_cnn = load_model('tfeb_aug_dropout_vgg16_freezall.h5')
tl_img_aug_finetune_cnn = load_model('tfeb_aug_dropout_vgg16_unfreezblock_4and5.h5')

# model.save('tfeb_aug_dropout.h5') #basic
# model.save('tfeb_aug_dropout_vgg16_freezall.h5') #transfer freez_all
# model.save('tfeb_aug_dropout_vgg16_unfreezblock_4and5.h5') #transfer freez5 and 4


# load other configurations
IMG_DIM = (150, 150)
input_shape = (150, 150, 3)
num2class_label_transformer = lambda l: ['Cytosole' if x == 0 else 'Nuclues' for x in l]
class2num_label_transformer = lambda l: [0 if x == 'Cytosole' else 1 for x in l]

# load VGG model for bottleneck features
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                  input_shape=input_shape)
output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)
vgg_model.trainable = False

def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features


'''
test labales
'''
IMG_DIM = (150, 150)
os.chdir("/data/kanferg/Images/TFeb_data_base_for_classification_model_2021")
test_files = glob.glob('test_data/*')
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
#test_files[0].split('/')[1].split('_')[0].strip()
test_labels = [fn.split('/')[1].split('_')[0].strip() for fn in test_files]
#test_labels = [fn.split('/')[1].split('.')[0].strip() for fn in test_files]

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
test_labels_enc = class2num_label_transformer(test_labels)

print('Test dataset shape:', test_imgs.shape)
print(test_labels[0:5], test_labels_enc[0:5])

#Test dataset shape: (1600, 150, 150, 3)
#['Cytosole', 'Nuclues', 'Nuclues', 'Cytosole', 'Cytosole'] [0, 1, 1, 0, 0]

'''
Model 1: Basic CNN Performance
'''
predictions = basic_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)


# meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions, 
#                                       classes=list(set(test_labels)))
#the meu is not working

meu.get_metrics(test_labels, predictions)
'''
Accuracy: 0.7738
Precision: 0.777
Recall: 0.7738
F1 Score: 0.7731
'''

'''
model 2: tl_img_aug_cnn
'''
predictions = tl_img_aug_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)


# meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions, 
#                                       classes=list(set(test_labels)))
#the meu is not working

meu.get_metrics(test_labels, predictions)
'''
Accuracy: 0.66
Precision: 0.6623
Recall: 0.66
F1 Score: 0.6588
'''

'''
model 3: tl_img_aug_finetune_cnn
'''
predictions = tl_img_aug_finetune_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)


# meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions, 
#                                       classes=list(set(test_labels)))
#the meu is not working

meu.get_metrics(test_labels, predictions)
'''
Accuracy: 0.7725
Precision: 0.7749
Recall: 0.7725
F1 Score: 0.772
'''















