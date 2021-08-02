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
#### What is One Hot Encoding?
One hot encoding is one method of converting data to prepare it for an algorithm and get a better prediction. With one-hot, converts each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. Each integer value is represented as a binary vector. All the values are zero, and the index is marked with a 1.
![example_one_hot](https://user-images.githubusercontent.com/36865654/127770957-a5cf936e-c2a5-4da2-ac04-2a43a20a8f19.png)

In this project, masks are converted to tensors with one hot encoding. One hot encoding is inside the one_hot_encoder method.

```
  result = np.zeros((data.shape[0],data.shape[1],n_class),dtype=np.int)
  encoded_labels = [[0,1], [1,0]]
  
  for i, encoded_labels[0] in enumerate(np.unique(data)):
       result[:,:,i][data==encoded_labels[0]] = 1

  return result
```

## Model -> U-Net
In this project, U-Net is used as the model. 

#### U-Net
U-Net is an auto-encoder-decoder network designed for medical image segmentation. The industry also regards it as an FCN (fully connected network). It can be divided into two parts, down(encoder) and up(decoder). The main structure of down can be seen as conv followed by maxpool. The main structure of up is an upsample followed by conv.

To understand this problem, we must first understand the role of convolution. Take the simple CNN network for training digital recognition in the MINIST data set as an example. It abstracts a 28*28 image into a 0-9 vector.Convolution can be seen as feature extraction, It can extract the abstract concept of the input information.But Pool and Conv will lose spatial information. Among them, the spatial information is more seriously lost in the pool process. For image segmentation, spatial information is as important as abstract information. Since each secondary pool will severely lose spatial information, that is to say, there is more spatial information between maxpool than later. So Unet proposed to connect the down feature to the corresponding up.

##### U-Net Structure
![example_unet](https://user-images.githubusercontent.com/36865654/127876280-165d40f6-f2e0-4a79-8ef5-60407d14bfe1.png)

In the picture input image tile, it is the training data. Except that the first layer is two conv, other layers can be seen as maxpool followed by two conv. Most of the conv in Unet exists in the form of two convs. One can be customized first for convenience.

##### DoubleConv Class
```
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
 
```

##### Down Path
It's structure is a maxpoolOne by onedouble_conv
```
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
```
##### Up Path
The main structure of U-Net's up path is upsample. But ConvTranspose2d can also be used instead of upsample. The code below gives two options.

```
class Up(nn.Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
 ```
##### The wheel has been built, then let's implement U-Net and let it run.
```
class FoInternNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FoInternNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.out = torch.sigmoid
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.out(logits)
        return logits
```
## Train
In train.py script, parameters and file paths are defined first.
```
######### PARAMETERS #########
valid_size = 0.3
test_size  = 0.1
batch_size = 4
epochs = 20
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################

# MODEL PATH
modelPath = os.path.join(MODEL_DIR)

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list
test_label_path_list = mask_path_list

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list
valid_label_path_list = mask_path_list

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list
train_label_path_list = mask_path_list


# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size
```
![train1](https://user-images.githubusercontent.com/36865654/127883894-ee8581d1-5c2e-453d-b912-4103e6fbe55d.png) ![train2](https://user-images.githubusercontent.com/36865654/127883907-9497e4e3-906c-428f-9c55-39d3a3b71732.png)

The model is called.
```
model = FoInternNet(n_channels=3, n_classes=2, bilinear=True)
```
Define loss function and optimizer.
```
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```
Then, the images are converted to tensors with the tensorize_image and tensorize_mask methods in the preprocess.py script start training with the model and training and validation loss in each epoch is printed.
```
# TRAINING THE NEURAL NETWORK
for epoch in tqdm.tqdm(range(epochs)):
    running_loss = 0
    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

        optimizer.zero_grad()

        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(ind)
        
        if ind == steps_per_epoch-1:
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                val_losses.append(val_loss)
                break

            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
```
The model is saved in the model path.
```
torch.save(model, modelPath)
```
The results are plotted graphically.
```
loss_train = [float(i)/sum(train_losses) for i in train_losses]
loss_val = [float(i)/sum(val_losses) for i in val_losses]
epochs = list(range(1,epochs+1,1))
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
![result_without_aug](https://user-images.githubusercontent.com/36865654/127886888-3a48c621-5364-447a-8946-64953d45b75f.png)

And finally the predicted masks are drawn.
```
# Draw predicted masks
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
```
##### An example of a predicted mask:

![predicted_mask_example](https://user-images.githubusercontent.com/36865654/127888016-c9696a86-c5dd-4d5e-9b09-9bf42b20fe5e.png)















