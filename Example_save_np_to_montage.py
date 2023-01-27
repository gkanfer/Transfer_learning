#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:26:57 2021

@author: kanferg
"""
path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/Cytosole"
os.chdir(path_orig)
image_name = glob.glob("*.png")

image_size=2
row_num=3
col_num=3
base_img=np.zeros((image_size*2*row_num,image_size*2*col_num))
strt=[1]
for row in range(0,row_num-1):
    tmp=strt[row]+2+image_size
    strt.append(tmp)
ends=[image_size+1]
for row in range(0,row_num-1):
    tmp=ends[row]+image_size+2
    ends.append(tmp)
tmp_np=np.ones((image_size,image_size))    
for col in range(col_num):
    for row in range(row_num):
        base_img[strt[col]:ends[col],strt[row]:ends[row]] = tmp_np


        
   