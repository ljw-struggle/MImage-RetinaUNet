# -*- coding: utf-8 -*-
import h5py
import numpy as np

def load_hdf5(in_file):
  with h5py.File(in_file, 'r') as file:
    return file['data'][()]

def write_hdf5(data, out_file):
  with h5py.File(out_file, 'w') as file:
    file.create_dataset('data', data=data, dtype=data.dtype)

def recompose_overlap(preds, patch_h, patch_w, stride_h, stride_w, n_h, n_w, num_image=20, full_height=584, full_width=565):
    full_prob = np.zeros((num_image, patch_h+stride_h*(n_h-1), patch_w+stride_w*(n_w-1), 1))
    full_sum = np.zeros((num_image, patch_h+stride_h*(n_h-1), patch_w+stride_w*(n_w-1), 1))

    for i in range(num_image):
        for h in range(n_h):
            for w in range(n_w):
                full_prob[i, h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[i*n_h*n_w+h*n_w+w]
                full_sum[i, h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1

    return (full_prob/full_sum)[:, full_height, full_width, :]

def recompose(preds, patch_h, patch_w, n_h, n_w, num_image=20, full_height=584, full_width=565):
    full_prob = np.zeros((num_image, patch_h*n_h, patch_w*n_w, 1))

    for i in range(num_image):
        for h in range(n_h):
            for w in range(n_w):
                full_prob[i, h*patch_h:(h+1)*patch_h, w*patch_w:(w+1)*patch_w, :]= preds[i*n_h*n_w+h*n_w+w]

    return full_prob[:, 0:full_height, 0:full_width, :]
