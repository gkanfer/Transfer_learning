#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:26:57 2021

@author: kanferg
"""
path_orig="/data/kanferg/Images/TFeb_data_base_for_classification_model_2021/Cytosole"
os.chdir(path_orig)
image_name = glob.glob("*.png")

image_size=401
base_img=np.zeros((image_size+1,image_size+1))# with streds
row_num=4
col_num=4
for row in range(1,row_num-1):
    base_img=np.concatenate([base_img,base_img],axis=1)
for col in range(1,col_num-1):
    base_img=np.concatenate([base_img,base_img],axis=0)
count=0
strt=[1]
for row in range(0,row_num-1):
    tmp=strt[row]+image_size
    strt.append(tmp)
ends=[image_size]
for row in range(0,row_num-1):
    tmp=ends[row]+image_size
    ends.append(tmp)
    
tmp_np=np.ones((image_size,image_size))    
for col in range(col_num):
    for row in range(row_num):
        base_img[strt[row]:ends[row],strt[col]:ends[col]] = tmp_np
    
   


        
   