from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.ticker as mticker
import torch
from matplotlib import pyplot as plt
import cv2
import tqdm
from constant import *
import numpy as np 
from model import FoInternNet
cuda = True
n_classes = 2 
input_shape= (224,224)


image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
test_input_path_list = image_path_list

modelPath = os.path.join(MODEL_DIR)
predictpath = os.listdir(PREDICT_DIR)

model = FoInternNet(n_channels=3, n_classes=2, bilinear=True)
model.cuda()
model = torch.load(modelPath)
model.eval()

#Write predict masks
for i in tqdm.tqdm(range(len(test_input_path_list))):
    batch_test = test_input_path_list[i:i+1]
    test_input = tensorize_image(batch_test, input_shape, cuda)
    outs = model(test_input)
    out=torch.argmax(outs,axis=1)
    out_cpu = out.cpu()
    outputs_list=out_cpu.detach().numpy()
    mask=np.squeeze(outputs_list,axis=0)
    mask = mask * 100
    predict_mask_path = os.path.join(PREDICT_MASK_DIR, batch_test[0] + ".png")
    predict_mask_path = predict_mask_path.replace('image','predict_mask')

    cv2.imwrite(predict_mask_path, mask.astype(np.uint8))

mask_list = os.listdir(PREDICT_MASK_DIR)

#Write predict masks to image
for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
    mask_name_without_ex = mask_name.split('.')[0]

    # Access required folders
    predict_mask_path = os.path.join(PREDICT_MASK_DIR, mask_name)
    image_path = os.path.join(IMAGE_DIR, mask_name_without_ex+'.jpg')
    predict_path = os.path.join(PREDICT_DIR, mask_name)

    # Read predict mask and corresponding original image
    mask  = cv2.imread(predict_mask_path, 0).astype(np.uint8)
    mask = mask / 100
    image = cv2.imread(image_path).astype(np.uint8)
    image=cv2.resize(image,(224, 224))

    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
    cpy_image = image.copy()
    image[mask==1, :] = (255, 0, 125)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)

    # Write output image into predict_masks folder
    cv2.imwrite(predict_path, opac_image)

