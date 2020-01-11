# -*- coding: utf-8 -*-
import cv2
import random
from utils.utils import *

num_image, height, width, channel = 20, 584, 565, 3

class loader(object):
    def __init__(self):
        pass

    @classmethod
    def get_data_training(self, original_image_path, ground_truth_path, patch_height, patch_width, num_patch, inside_FOV):
        train_imgs_original = load_hdf5(original_image_path)
        train_masks = load_hdf5(ground_truth_path)
        train_imgs = self.preprocess(train_imgs_original)
        train_masks = train_masks/255.
        train_imgs = train_imgs[:,:,9:574,:] # cut bottom and top so now it is 565*565
        train_masks = train_masks[:,:,9:574,:] # cut bottom and top so now it is 565*565
        patches_imgs_train, patches_masks_train = self.extract_random(train_imgs,train_masks,patch_height,patch_width,num_patch,inside_FOV)
        return patches_imgs_train, patches_masks_train

    @classmethod
    def get_data_testing(self, original_image_path, ground_truth_path, patch_height, patch_width):
        test_imgs_original = load_hdf5(original_image_path)
        test_masks = load_hdf5(ground_truth_path)
        test_imgs = self.preprocess(test_imgs_original)
        test_masks = test_masks/255.
        test_imgs = test_imgs[:,:,:,:]
        test_masks = test_masks[:,:,:,:]
        test_imgs = self.paint_border(test_imgs,patch_height,patch_width)
        test_masks = self.paint_border(test_masks,patch_height,patch_width)
        patches_imgs_test = self.extract_ordered(test_imgs,patch_height,patch_width)
        patches_masks_test = self.extract_ordered(test_masks,patch_height,patch_width)
        return patches_imgs_test, patches_masks_test

    @classmethod
    def get_data_testing_overlap(self, original_image_path, ground_truth_path, patch_height, patch_width, stride_height, stride_width):
        test_imgs_original = load_hdf5(original_image_path)
        test_masks = load_hdf5(ground_truth_path)
        test_imgs = self.preprocess(test_imgs_original)
        test_masks = test_masks/255.
        test_imgs = test_imgs[:,:,:,:]
        test_masks = test_masks[:,:,:,:]
        test_imgs = self.paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
        patches_imgs_test = self.extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
        return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks

    @staticmethod
    def preprocess(data):
        train_imgs = rgb2gray(data)

        # 1\ Normalization
        imgs_std = np.std(train_imgs)
        imgs_mean = np.mean(train_imgs)
        imgs_normalized = (train_imgs - imgs_mean) / imgs_std
        for i in range(train_imgs.shape[0]):
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                        np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255

        # 2\ CLAHE Equalized
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = np.empty(imgs_normalized.shape)
        for i in range(imgs_normalized.shape[0]):
            imgs_equalized[i, 0] = clahe.apply(np.array(imgs_normalized[i, 0], dtype=np.uint8))
        # histogram equalization
        # imgs_equalized = np.empty(imgs_normalized.shape)
        # for i in range(imgs_normalized.shape[0]):
        #     imgs_equalized[i, 0] = cv2.equalizeHist(np.array(imgs_normalized[i, 0], dtype=np.uint8))

        # 3\ Adjust Gamma
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        new_imgs = np.empty(imgs_equalized.shape)
        for i in range(imgs_equalized.shape[0]):
            new_imgs[i, 0] = cv2.LUT(np.array(imgs_equalized[i, 0], dtype=np.uint8), table)

        train_imgs = new_imgs/255.

        return train_imgs

    @staticmethod
    def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
        def is_patch_inside_FOV(x, y, img_w, img_h, patch_h): # check if the patch is fully contained in the FOV
            x_ = x - int(img_w / 2) # origin (0,0) shifted to image center
            y_ = y - int(img_h / 2) # origin (0,0) shifted to image center
            R_inside = 270 - int(patch_h * np.sqrt(
                2.0) / 2.0) # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
            radius = np.sqrt((x_ * x_) + (y_ * y_))
            return True if radius < R_inside else False

        patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
        patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
        img_h = full_imgs.shape[2] # height of the full image
        img_w = full_imgs.shape[3] # width of the full image
        patch_per_img = int(N_patches/full_imgs.shape[0]) # N_patches equally divided in the full images
        iter_tot = 0 # iter over the total number rof patches (N_patches)
        for i in range(full_imgs.shape[0]): # loop over the full images
            k=0
            while k <patch_per_img:
                x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
                y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
                if inside==True:
                    if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
                        continue
                patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
                patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
                patches[iter_tot]=patch
                patches_masks[iter_tot]=patch_mask
                iter_tot +=1
                k+=1
        return patches, patches_masks

    @staticmethod
    def extract_ordered(full_imgs, patch_h, patch_w):
        img_h = full_imgs.shape[2]
        img_w = full_imgs.shape[3]
        N_patches_h = int(img_h/patch_h) # round to lowest int
        N_patches_w = int(img_w/patch_w) # round to lowest int
        if (img_h%patch_h != 0) or (img_h%patch_h != 0):
            print('warning: ' +str(N_patches_h) +' patches in height, with about ' +str(img_h%patch_h) +' pixels left over')
        N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
        patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
        iter_tot = 0 # iter over the total number of patches (N_patches)
        for i in range(full_imgs.shape[0]): # loop over the full images
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    patch = full_imgs[i,:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                    patches[iter_tot]=patch
                    iter_tot +=1
        return patches

    @staticmethod
    def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
        img_h = full_imgs.shape[1]
        img_w = full_imgs.shape[2]
        N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  # // --> division between integers
        N_patches_tot = N_patches_img*full_imgs.shape[0]
        patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
        iter_tot = 0   # iter over the total number of patches (N_patches)
        for i in range(full_imgs.shape[0]):  # loop over the full images
            for h in range((img_h-patch_h)//stride_h+1):
                for w in range((img_w-patch_w)//stride_w+1):
                    patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                    patches[iter_tot]=patch
                    iter_tot +=1   # total
        return patches

    @staticmethod
    def paint_border(data, patch_h, patch_w): # Extend the full images because patch division is not exact
        img_h=data.shape[1]
        img_w=data.shape[2]
        new_img_h = img_h if (img_h%patch_h)==0 else ((int(img_h)/int(patch_h))+1)*patch_h
        new_img_w = img_w if (img_w%patch_w)==0 else ((int(img_w)/int(patch_w))+1)*patch_w
        new_data = np.zeros((data.shape[0],data.shape[1],new_img_h,new_img_w))
        new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
        return new_data

    @staticmethod
    def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
        img_h = full_imgs.shape[1]
        img_w = full_imgs.shape[2]
        leftover_h = (img_h-patch_h)%stride_h # leftover on the h dim
        leftover_w = (img_w-patch_w)%stride_w # leftover on the w dim
        if (leftover_h != 0): # change dimension of img_h
            tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
            tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
            full_imgs = tmp_full_imgs
        if (leftover_w != 0): # change dimension of img_w
            tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
            tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
            full_imgs = tmp_full_imgs
        return full_imgs