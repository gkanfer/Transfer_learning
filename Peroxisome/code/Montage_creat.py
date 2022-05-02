#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function which vreat a montage from list of files
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
from PIL import Image
from skimage.exposure import rescale_intensity
from random import randint

def unique_rand(inicial, limit, total):
    data = []
    i = 0
    while i < total:
                number = randint(inicial, limit)
                if number not in data:
                    data.append(number)
                    i += 1
    return data


def show_image_adgust(image,low_prec,up_prec,np_return=True):
    """
    image= np array 2d
    low/up precentile border of the image
    """
    percentiles = np.percentile(image, (low_prec, up_prec))
    scaled_ch1 = exposure.rescale_intensity(image, in_range=tuple(percentiles))
    if np_return:
        return scaled_ch1
    else:
        PIL_scaled_ch1 = Image.fromarray(np.uint16(scaled_ch1))
        PIL_scaled_ch1.show()
 #show_image_adgust(Ch2,1,99)



def make_montage(image_name,row_num=3,col_num=3,image_size=401):
    '''
    

    Parameters
    ----------
    image_name : str,
        list of image files name.
    row_num : int, optional
        DESCRIPTION. The default is 3.
    col_num : int, optional
        DESCRIPTION. The default is 3.
    image_size : int, optional
        DESCRIPTION. The default is 401.

    Returns
    -------
    base_img : np array in the the size of the matrix
        row col needs to be equale to rows.

    '''
    base_img=np.zeros((image_size*2*row_num,image_size*2a*col_num))
    strt=[1]
    for row in range(0,row_num-1):
        tmp=strt[row]+2+image_size
        strt.append(tmp)
    ends=[image_size+1]
    for row in range(0,row_num-1):
        tmp=ends[row]+image_size+2
        ends.append(tmp)
    for col in range(col_num):
        for row in range(row_num):
            ret,img=cv2.imreadmulti(image_name[unique_rand(1, len(image_name), 1)[0]], [], cv2.IMREAD_ANYDEPTH)  
            img=np.array(img).reshape(image_size,image_size)
            base_img[strt[col]:ends[col],strt[row]:ends[row]] = img
    return base_img


path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/Cytosole"
path_out="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp"
os.chdir(path_orig)
image_name = glob.glob("*.png")

img_montage = make_montage(image_name=image_name,row_num=3,col_num=3,image_size=401)

#PIL_image = Image.fromarray(np.uint8(img_montage))
#PIL_image.show()

#save montages to temp 
for i in range(5):
    os.chdir(path_orig)
    img_montage = make_montage(image_name=image_name,row_num=3,col_num=3,image_size=401)
    PIL_image = Image.fromarray(np.uint8(img_montage))
    os.chdir(path_out)
    PIL_image.save("cytosole_"+str(i)+".png")
    
#save nuclus
path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/Nuclues"
path_out="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/temp"
os.chdir(path_orig)
image_name = glob.glob("*.png")

img_montage = make_montage(image_name=image_name,row_num=3,col_num=3,image_size=401)

#PIL_image = Image.fromarray(np.uint8(img_montage))
#PIL_image.show()

#save montages to temp 
for i in range(5):
    os.chdir(path_orig)
    img_montage = make_montage(image_name=image_name,row_num=3,col_num=3,image_size=401)
    PIL_image = Image.fromarray(np.uint8(img_montage))
    os.chdir(path_out)
    PIL_image.save("nuc"+str(i)+".png")