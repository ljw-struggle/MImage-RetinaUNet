# -*- coding: utf-8 -*-
import h5py
import numpy as np
from PIL import Image

def load_hdf5(in_file):
  with h5py.File(in_file, 'r') as file:
    return file['data'][()]

def write_hdf5(data, out_file):
  with h5py.File(out_file, 'w') as file:
    file.create_dataset('data', data=data, dtype=data.dtype)

def group_images(data, per_row):
    data = np.transpose(data,(0,2,3,1))
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

def visualize(data,filename):
    if data.shape[2] == 1:
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))
    else:
        img = Image.fromarray((data*255).astype(np.uint8))
    img.save(filename + '.png')

def recompose_overlap(preds, patch_h, patch_w, stride_h, stride_w, n_h, n_w, num, full_height=584, full_width=565):
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
    return final_avg

def recompose(preds, patch_h, patch_w, n_h, n_w, num, full_height=584, full_width=565):
    full_recomp = np.empty((N_full_imgs,data.shape[1],N_h*patch_h,N_w*patch_w))
    while (s<data.shape[0]):
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]=data[s]
        full_recomp[k]=single_recon
    return full_recomp