import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from tqdm import tqdm
import glob
import numpy as np
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

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
from utils import Taining_data_orgenizer as orgenizer 
from utils import model_builder as mb

import tensorflow as tf
from tensorboard.plugins.x import api as hp
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
path_model = '/data/kanferg/Images/Pex_project/Transfer_learning/models'
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
                 imbalance_train = imbalance_train, imbalance_val = imbalance_val)
train_imgs_scaled, validation_imgs_scaled,train_labels_enc,validation_labels_enc,train_imgs,validation_imgs,report = model_build.build_image__sets()

x_train = train_imgs_scaled
y_train = train_labels_enc
x_test  = validation_imgs_scaled
y_test = validation_labels_enc
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256, 512]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5, 0.6))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd','rmsprop']))
HP_L2 = hp.HParam('l2 regularizer', hp.RealInterval(.001,.01))

METRIC_ACCURACY = 'accuracy'
os.chdir(path_model)
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER,HP_L2],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
#VGG16
def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(150,150,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax),
    ])
    model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
    )

    model.fit(x_train, y_train, epochs=10) 
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=2)

    session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for l2 in (HP_L2.domain.min_value, HP_L2.domain.max_value):
      for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            HP_NUM_UNITS: num_units,
            HP_DROPOUT: dropout_rate,
            HP_L2: l2,
            HP_OPTIMIZER: optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1