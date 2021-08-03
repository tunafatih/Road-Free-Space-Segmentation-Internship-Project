from model import FoInternNet
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
import gc


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

# CALL MODEL

model = FoInternNet(n_channels=3, n_classes=2, bilinear=True)


# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

val_losses=[]
train_losses=[]
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
            
torch.save(model, modelPath)


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

        
    









