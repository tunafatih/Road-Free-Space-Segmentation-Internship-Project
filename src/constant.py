import os


# Path to jsons
JSON_DIR = 'D:\Ford_Intern\intern-p1\data\jsons'

# Path to mask
MASK_DIR  = 'D:\Ford_Intern\intern-p1\data\masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = 'D:\Ford_Intern\intern-p1\data\masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = 'D:\Ford_Intern\intern-p1\data\images'

# Path to model
MODEL_DIR = 'D:\Ford_Intern\intern-p1/src/model.pt'

# Path to predict
PREDICT_DIR = 'D:\Ford_Intern\intern-p1\data\predicts'

# Path to predict masks
PREDICT_MASK_DIR  = 'D:\Ford_Intern\intern-p1\data\predict_masks'

# Path to data augmentation
AUGMENTATION_DIR = r'D:\Ford_Intern\intern-p1\data\augmentations'

# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = True

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2