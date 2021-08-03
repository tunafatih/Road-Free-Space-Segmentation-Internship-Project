import os, tqdm, cv2
import numpy as np
from constant import *

mask_list = os.listdir(PREDICT_MASK_DIR)


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
