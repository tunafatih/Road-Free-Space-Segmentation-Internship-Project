# Road Free Space Segmentation Internship Project
This repository contains the detecting of freespace with semantic segmentation using PyTorch.

## What is Freespace?

Freespace is the drivable area of the highway.
![example_freespace](https://user-images.githubusercontent.com/36865654/127042047-f071a48e-3190-461e-80f0-72f46634e1e8.jpg)

 ## Data - Images and Jsons
 
 Data is needed for semantic segmentation. The data is labeled images in this project. Labeled images contain freespace, lines, traffic signs, etc. This project focus on just freespace. This project's data is images and jsons. Images consist of photos taken on highways. Jsons consist of values labeled in images.
 
 #### Image 
 ![image](https://user-images.githubusercontent.com/36865654/127346157-34c2e96d-ec7e-4f48-a2bb-85c2bbb5f5f2.jpg)
 #### Json
 ![json](https://user-images.githubusercontent.com/36865654/127346167-9023a8ff-72d9-4143-9d57-c6950057bb72.png)

 ## Constant
 constant.py is a script with file paths and some values.
 ``` 
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
 ```
## Json To Mask
json2mask.py is a script that convert JSON to mask. The jsons in the jsons folder are selected with a loop and when the classTitle is freespace, the values are drawn with the fillypolly method of the opencv library.
```
# For every json file
for json_name in tqdm.tqdm(json_list):

    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)

    # Create an empty mask whose size is the same as the original image's size
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':
            # Extract exterior points which is a point list that contains
            # every edge of polygon and fill the mask with the array.
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)

    # Write mask image into MASK_DIR folder
    cv2.imwrite(mask_path, mask.astype(np.uint8))
```
#### An example of a mask
![example_mask](https://user-images.githubusercontent.com/36865654/127341903-e65f00da-47e9-47e7-bd0c-ebeaa4f2b42e.png)

## Mask on Image
mask_on_image.py is a script that puts the mask on the image. Masks in the mask folder are selected with a loop and put into the corresponding image. New images are written to another folder.

```
image[mask==1, :] = (69, 190, 121)

```
The color of freespace can be changed with the values in this line.

#### After mask_on_image.py script -> Masked Image
![example_masked_image](https://user-images.githubusercontent.com/36865654/127345781-d2a4a42b-6c26-4552-853c-77abc52a260b.png)














