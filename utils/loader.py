# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
from utils.utils import *
from PIL import Image

class loader(object):
    @classmethod
    def get_data_training(self, original_image_path, ground_truth_path, border_mask_path,
                          patch_height, patch_width, num_patch, inside_FOV=False):
        original_images = load_hdf5(original_image_path) # shape = (-1, 584, 565, 3)
        ground_truths = load_hdf5(ground_truth_path) # shape = (-1, 584, 565, 1)
        masks = load_hdf5(border_mask_path) # shape = (-1, 584, 565, 1)

        # 1\ Processing
        processed_images = self.preprocess(original_images) # shape = (-1, 584, 565, 1)
        ground_truths = ground_truths # shape = (-1, 584, 565, 1)
        masks = masks # shape = (-1, 584, 565, 1)

        # 2\ Divide to patches.
        num_patch_per_img = int(num_patch / processed_images.shape[0])
        processed_image_patches, ground_truth_patches = [], []
        for i in range(processed_images.shape[0]):
            for k in range(num_patch_per_img):
                y_ = random.randint(0, 584 - patch_height)
                x_ = random.randint(0, 565 - patch_width)
                if inside_FOV:
                    if masks[i, y_, x_, 0]==0 or masks[i, y_+patch_height, x_+patch_width, 0]==0 or \
                       masks[i, y_, x_, 0]==0 or masks[i, y_+patch_height, x_+patch_width, 0]==0:
                        continue
                processed_image_patch = processed_images[i, y_:y_ + patch_height, x_:x_ + patch_width, :]
                ground_truth_patch = ground_truths[i, y_:y_ + patch_height, x_:x_ + patch_width, :]
                processed_image_patches.append(processed_image_patch)
                ground_truth_patches.append(ground_truth_patch)

        return np.array(processed_image_patches), np.array(ground_truth_patches)

    @classmethod
    def get_data_testing(self, original_image_path, patch_height, patch_width):
        original_images = load_hdf5(original_image_path) # shape = (-1, 584, 565, 3)

        # 1\ Processing
        processed_images = self.preprocess(original_images) # shape = (-1, 584, 565, 1)

        # 2\ Paint Border.
        new_height = int(np.ceil(584/patch_height)*patch_height)
        new_width = int(np.ceil(565/patch_width)*patch_width)
        new_images = np.zeros((processed_images.shape[0], new_height, new_width, 1))
        new_images[:, 0:584, 0:565, :] = processed_images

        # 3\ Divide to patches.
        processed_image_patches = []
        num_sample = new_images.shape[0]
        num_patch_height, num_patch_width = int(new_height / patch_height), int(new_width / patch_width)
        for i in range(num_sample):
            for h in range(num_patch_height):
                for w in range(num_patch_width):
                    processed_image_patch = \
                        new_images[i, h*patch_height:(h+1)*patch_height, w*patch_width:(w+1)*patch_width, :]
                    processed_image_patches.append(processed_image_patch)

        return np.array(processed_image_patches), num_patch_height, num_patch_width, original_images.shape[0]

    @classmethod
    def get_data_testing_overlap(self, original_image_path, patch_height, patch_width, stride_height, stride_width):
        original_images = load_hdf5(original_image_path) # shape = (-1, 584, 565, 3)

        # 1\ Processing
        processed_images = self.preprocess(original_images) # shape = (-1, 584, 565, 1)

        # 2\ Paint Border.
        new_height = int(np.ceil((584 - patch_height) / stride_height) * stride_height + patch_height)
        new_width = int(np.ceil((565 - patch_width) / stride_width) * stride_width + patch_width)
        new_images = np.zeros((processed_images.shape[0], new_height, new_width, 1))
        new_images[:, 0:584, 0:565, :] = processed_images

        # 3\ Divide to patches.
        processed_image_patches = []
        num_sample = new_images.shape[0]
        num_patch_height, num_patch_width = int((new_height-patch_height)/stride_height + 1), \
                                            int((new_width-patch_width)/stride_width + 1)
        for i in range(num_sample):
            for h in range(num_patch_height):
                for w in range(num_patch_width):
                    processed_image_patch = new_images[i, h*stride_height:h*stride_height+patch_height,
                                                       w*stride_width:w*stride_width+patch_width, :]
                    processed_image_patches.append(processed_image_patch)
        return np.array(processed_image_patches), num_patch_height, num_patch_width, original_images.shape[0]

    @staticmethod
    def preprocess(data, gamma=1.2):
        # 1\ Change RGB to GRAY
        train_images = data[:, :, :, 0:1] * 0.299 + data[:, :, :, 1:2] * 0.587 + data[:, :, :, 2:3] * 0.114

        # 2\ Normalization
        images_std = np.std(train_images)
        images_mean = np.mean(train_images)
        images_normalized = (train_images - images_mean) / images_std

        for i in range(train_images.shape[0]):
            images_normalized[i] = ((images_normalized[i] - np.min(images_normalized[i])) /
                                    (np.max(images_normalized[i]) - np.min(images_normalized[i]))) * 255

        # 3\ C-L-A-H-E Equalization
        images_equalized = np.empty(images_normalized.shape)
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(images_normalized.shape[0]):
            images_equalized[i,:,:,0] = CLAHE.apply(np.array(images_normalized[i,:,:,0], dtype=np.uint8))

        # 4\ Adjust Gamma
        images_lut = np.empty(images_equalized.shape)
        inverse_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in np.arange(256)]).astype('uint8')
        for i in range(images_equalized.shape[0]):
            images_lut[i,:,:, 0] = cv2.LUT(np.array(images_equalized[i,:,:, 0], dtype=np.uint8), table)

        # 5\ Change [0, 255] to [0, 1].
        train_images = images_lut/255

        return train_images