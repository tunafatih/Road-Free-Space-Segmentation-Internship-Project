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
json2mask.py is a script that converts JSON to mask. The jsons in the jsons folder are selected with a loop and when the classTitle is freespace, the values are drawn with the fillypolly method of the opencv library.
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

## Preprocess
Preprocess.py is a script that converts images and masks to tensors.

##### What are Tensors?
A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array. A vector is a one-dimensional or first order tensor and a matrix is a two-dimensional or second order tensor.
##### tensorize_image method
The script's tensorize_image method converts images to tensors. The image is resized according to the defined shape. The images in the image folder are selected by loop and the torchlike_data method changes the input structure according to the pytorch input structure and images convert from list structure to torch tensor.
 ```
    # Create empty list
    local_image_list = []

    # For each image
    for image_path in image_path_list:

        # Access and read image
        image = cv2.imread(image_path)

        # Resize the image according to defined shape
        image = cv2.resize(image, output_shape)

        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)

        # Add into the list
        local_image_list.append(torchlike_image)

    # Convert from list structure to torch tensor
    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()
    
    return torch_image
```
##### torchlike_data method
```
# Obtain channel value of the input
    n_channels = data.shape[2]

    # Create and empty image whose dimension is similar to input
    torchlike_data_output = np.empty((n_channels, data.shape[0], data.shape[1]))

    # For each channel

    for ch in range(n_channels):
        torchlike_data_output[ch] = data[:,:,ch]

    return torchlike_data_output
```
##### tensorize_mask method
The script's tensorize_mask method converts masks to tensors. The mask is resized according to the defined shape. The masks in the mask folder are selected by loop and the one_hot_encoder method applies one-hot encoding to image. torchlike_data method changes the input structure according to the pytorch input structure and images convert from list structure to torch tensor.
```
# Create empty list
    local_mask_list = []

    # For each masks
    for mask_path in mask_path_list:

        # Access and read mask
        mask = cv2.imread(mask_path, 0)

        # Resize the mask according to defined shape
        mask = cv2.resize(mask, output_shape)

        # Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, n_class)

        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)


        local_mask_list.append(torchlike_mask)

    mask_array = np.array(local_mask_list, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    
    return torch_mask

```










