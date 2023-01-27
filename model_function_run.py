#run the Training data orgenizer and model build from class

#import glob
#import numpy as np

import shutil
#import glob
#from sklearn.model_selection import train_test_split
import os
os.chdir('/data/kanferg/Images/Pex_project/Transfer_learning/code')
from utils import Taining_data_orgenizer as orgenizer 
from utils import model_builder as mb
#import pdb

path_input = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/input_sc_mix'
path_origen = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set'
label_A = 'norm'
label_B = 'pheno'
file_extention = 'png' 

path_builder = orgenizer.classification_data_orgenizer(path_input = path_input,path_origen = path_origen,label_A=label_A,label_B =label_B,
                                                       file_extention =file_extention)

path_builder.get_file_names_list()

statment_a, statment_b, train_files, validate_files, test_files = path_builder.split_traning_set_and_copy()
print(statment_a)
print(statment_b)

x = [file_name for file_name in train_files if 'pheno' in file_name]
len(x)
'''
A set datasets: (9217,)(1152,)(1153,)
B set datasets: (9217,)(1152,)(4952,)
'''

train_dir = os.path.join(path_origen, 'training_data')
val_dir = os.path.join(path_origen, 'validation_data')
test_dir = os.path.join(path_origen, 'test_data')

os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None


os.chdir(path_input)
for fn in train_files:
    shutil.copy(fn, train_dir)
    print('{}'.format(fn))

for fn in validate_files:
    shutil.copy(fn, val_dir)
    print('{}'.format(fn))

for fn in test_files:
    shutil.copy(fn, test_dir)
    print('{}'.format(fn))


'''
remove files smaler then 5k
find . -name "*.png" -type 'f' -size -5k -delete

'''




