#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove black adges and focous on mass from image

"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os, re
import cv2
import glob
from scipy import ndimage, misc
import re
import tifffile as tiff
import math
from PIL import Image, ImageDraw
from skimage.exposure import rescale_intensity
from random import randint
from numpy import asarray
from skimage import data,io
from skimage.filters import threshold_otsu, threshold_local

#threshold_otsu, threshold_adaptive, threshold_local
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import scipy as ndimage
from scipy.ndimage.morphology import binary_opening
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import data, util
from skimage.measure import label
from skimage.measure import perimeter
from skimage import measure
import os
import pandas as pd
from pandas import DataFrame
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage as ndi

'''
display reactangle
'''

path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/training_data"
path_out="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp"
os.chdir(path_orig)
image_name = glob.glob("*.png")

for i in range(0,100):
    os.chdir(path_orig)
    ret,img=cv2.imreadmulti(image_name[i], [], cv2.IMREAD_ANYDEPTH)  
    img=np.array(img)
    img=img.reshape(np.shape(img)[1],np.shape(img)[2])
    
    cy, cx = ndi.center_of_mass(img)
    w, h = cy, cx
    shape = [(w-100, h-100), (w+100, h+100)]
      
    # creating new Image object
    
      
    # create rectangle image
    os.chdir(path_out)
    PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
    img1 = ImageDraw.Draw(PIL_image)  
    img1.rectangle(shape, fill =  None  , outline ="red")
    PIL_image.show()
    PIL_image.save("rect"+str(i)+".png")
    

'''
crop image  from center
'''

path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/training_data"
path_out="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp"
os.chdir(path_orig)
image_name = glob.glob("*.png")

for i in range(0,100):
    os.chdir(path_orig)
    ret,img=cv2.imreadmulti(image_name[i], [], cv2.IMREAD_ANYDEPTH)  
    img=np.array(img)
    img=img.reshape(np.shape(img)[1],np.shape(img)[2])
    
    cy, cx = ndi.center_of_mass(img)
    cx=int(cx)
    cy=int(cy)
    img=img[cy-100:cy+100,cx-100:cx+100]
    
    # create rectangle image
    os.chdir(path_out)
    PIL_image = Image.fromarray(np.uint8(img))
    PIL_image.show()
    PIL_image.save("crop"+str(i)+".png")    



'''
crop image and rewrite image 
'''

path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/training_data"
path_out="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp"
os.chdir(path_orig)
image_name = glob.glob("*.png")


def save_crop(input_path,out_path,image_name=image_name,w=100,h=100):
    os.chdir(input_path)
    ret,img=cv2.imreadmulti(image_name, [], cv2.IMREAD_ANYDEPTH)  
    img=np.array(img)
    img=img.reshape(np.shape(img)[1],np.shape(img)[2])
    
    cy, cx = ndi.center_of_mass(img)
    cx=int(cx)
    cy=int(cy)
    img=img[cy-w:cy+h,cx-w:cx+h]
    
    # create rectangle image
    os.chdir(out_path)
    PIL_image = Image.fromarray(np.uint8(img))
    #PIL_image.show()
    PIL_image.save(image_name)    
    


#test
path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/training_data"
path_out="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp"
os.chdir(path_orig)
image_name = glob.glob("*.png")
for i in range(5):
    save_crop(input_path=path_orig,out_path=path_out,image_name=image_name[i],w=100,h=100) 
    print(i)


path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/training_data"
os.chdir(path_orig)
image_name = glob.glob("*.png")
for i in range(len(image_name)):
    save_crop(input_path=path_orig,out_path=path_orig,image_name=image_name[i],w=100,h=100) 
    print(i)
path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/test_data"
os.chdir(path_orig)
image_name = glob.glob("*.png")
for i in range(len(image_name)):
    save_crop(input_path=path_orig,out_path=path_orig,image_name=image_name[i],w=100,h=100) 
    print(i)
path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/validation_data"
os.chdir(path_orig)
image_name = glob.glob("*.png")
for i in range(len(image_name)):
    save_crop(input_path=path_orig,out_path=path_orig,image_name=image_name[i],w=100,h=100) 
    print(i)







