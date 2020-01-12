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

def recompose_overlap(preds, img_h, img_w, stride_h, stride_w):
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    N_full_imgs = preds.shape[0]//N_patches_img
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    k = 0
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1
    final_avg = full_prob/full_sum
    print(final_avg.shape)
    return final_avg

def recompose(data,N_h,N_w):
    N_pacth_per_img = N_w*N_h
    N_full_imgs = data.shape[0]/N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    full_recomp = np.empty((N_full_imgs,data.shape[1],N_h*patch_h,N_w*patch_w))
    k, s = 0, 0
    while (s<data.shape[0]):
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]=data[s]
                s+=1
        full_recomp[k]=single_recon
        k+=1
    return full_recomp