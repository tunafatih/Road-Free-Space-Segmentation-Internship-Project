from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,
)
import albumentations
import numpy as np
import cv2
import os
import glob 
import tqdm
import torch
from constant import *

image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))

# HorizontalFlip image
for image in tqdm.tqdm(image_path_list):
    image_name_without_ex = image.split('.')[0]
    image = cv2.imread(image)

    augmentation = albumentations.HorizontalFlip(p=1.0)
    data = {"image": image}
    augmented = augmentation(**data)
    augmentation_images_path = os.path.join(AUGMENTATION_DIR, image_name_without_ex + "_flip" + ".jpg")
    augmentation_images_path = augmentation_images_path.replace('image','augmentation_image')
    cv2.imwrite(augmentation_images_path,augmented["image"])

# HorizontalFlip mask
for mask in tqdm.tqdm(mask_path_list):
    mask_name_without_ex = mask.split('.')[0]
    mask = cv2.imread(mask)

    augmentation = albumentations.HorizontalFlip(p=1.0)
    data = {"image": mask}
    augmented = augmentation(**data)
    
    augmentation_masks_path = os.path.join(AUGMENTATION_DIR, mask_name_without_ex + "_flip" + ".png")
    augmentation_masks_path = augmentation_masks_path.replace('mask','augmentation_mask')
    cv2.imwrite(augmentation_masks_path,augmented["image"])

# ToGray image
for image in tqdm.tqdm(image_path_list):
    image_name_without_ex = image.split('.')[0]
    image = cv2.imread(image)

    augmentation = albumentations.ToGray(p=1)
    data = {"image": image}
    augmented = augmentation(**data)
    augmentation_images_path = os.path.join(AUGMENTATION_DIR, image_name_without_ex + "_Gray" + ".jpg")
    augmentation_images_path = augmentation_images_path.replace('image','augmentation_image')
    cv2.imwrite(augmentation_images_path,augmented["image"])

# ToGray mask
for mask in tqdm.tqdm(mask_path_list):
    mask_name_without_ex = mask.split('.')[0]
    mask = cv2.imread(mask)
    augmentation_masks_path = os.path.join(AUGMENTATION_DIR, mask_name_without_ex + "_Gray" + ".png")
    augmentation_masks_path = augmentation_masks_path.replace('mask','augmentation_mask')
    cv2.imwrite(augmentation_masks_path,mask)

# ChannelShuffle image
for image in tqdm.tqdm(image_path_list):
    image_name_without_ex = image.split('.')[0]
    image = cv2.imread(image)

    augmentation = albumentations.ChannelShuffle(p=1)
    data = {"image": image}
    augmented = augmentation(**data)
    augmentation_images_path = os.path.join(AUGMENTATION_DIR, image_name_without_ex + "_ChannelShuffle" + ".jpg")
    augmentation_images_path = augmentation_images_path.replace('image','augmentation_image')
    cv2.imwrite(augmentation_images_path,augmented["image"])

# ChannelShuffle mask
for mask in tqdm.tqdm(mask_path_list):
    mask_name_without_ex = mask.split('.')[0]
    mask = cv2.imread(mask)
    augmentation_masks_path = os.path.join(AUGMENTATION_DIR, mask_name_without_ex + "_ChannelShuffle" + ".png")
    augmentation_masks_path = augmentation_masks_path.replace('mask','augmentation_mask')
    cv2.imwrite(augmentation_masks_path,mask)

# MultiplicativeNoise image
for image in tqdm.tqdm(image_path_list):
    image_name_without_ex = image.split('.')[0]
    image = cv2.imread(image)

    augmentation = albumentations.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1)
    data = {"image": image}
    augmented = augmentation(**data)
    augmentation_images_path = os.path.join(AUGMENTATION_DIR, image_name_without_ex + "_MultiplicativeNoise" + ".jpg")
    augmentation_images_path = augmentation_images_path.replace('image','augmentation_image')
    cv2.imwrite(augmentation_images_path,augmented["image"])

# MultiplicativeNoise mask
for mask in tqdm.tqdm(mask_path_list):
    mask_name_without_ex = mask.split('.')[0]
    mask = cv2.imread(mask)
    augmentation_masks_path = os.path.join(AUGMENTATION_DIR, mask_name_without_ex + "_MultiplicativeNoise" + ".png")
    augmentation_masks_path = augmentation_masks_path.replace('mask','augmentation_mask')
    cv2.imwrite(augmentation_masks_path,mask)