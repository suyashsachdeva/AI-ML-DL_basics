
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

import cv2
from loss_function import Loss
from complete_model import OneShot
from utility_functions import preprocessing
from data_preprocessing import to_torch_and_split

from complete_model import MyModel
from data_preprocessing import load_data
from utility_functions import train_model


PATH = r"folder"
# Defining constants for training cation
EPOCHS = 10              # Number of training epochs
DATA_SIZE = 5000         # Total size of the training dataset
BATCH_SIZE = 100         # Batch size for training
VALID_DATA = 500         # Total size of the validation dataset
VALID_BATCH = 100        # Batch size for validation
IMG_SHAPE = 64  

train = preprocessing(PATH)
xtrain1, xtrain2, xtrain3, xvalid1, xvalid2, xvalid3 = to_torch_and_split(train)

# Calculate the number of steps per epoch based on training and validation batches
steps = len(xtrain1)  # Number of steps (batches) in training data
vstep = len(xvalid1)  # Number of steps (batches) in validation data

model = OneShot()
# Define the triplet loss function with pairwise distance
criterion = Loss()

# Set initial learning rate and decay factor for learning rate scheduling
learning_rate = 1e-4
decay = 0.9        # Image dimensions (assumes square images of shape 64x64)


for epoch in trange(EPOCHS):  # Iterate through epochs
    lss = 0  # Initialize training loss for the epoch
    vls = 0  # Initialize validation loss for the epoch
    
    # Decay learning rate over time
    learning_rate = learning_rate / (1 + epoch * decay)
    
    # Optimizer with the updated learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop over batches
    for step, (xtr1, xtr2, xtr3) in enumerate(zip(xtrain1, xtrain2, xtrain3)):
        a = model(xtr1)  # Anchor output
        p = model(xtr2)  # Positive output
        n = model(xtr3)  # Negative output
        
        # Compute triplet loss
        loss = criterion(a, p, n)
        lss += loss  # Accumulate training loss
        
        # Zero the gradients, backpropagate, and update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop over batches
    for step, (xval1, xval2, xval3) in enumerate(zip(xvalid1, xvalid2, xvalid3)):
        a = model(xval1)  # Anchor output for validation
        p = model(xval2)  # Positive output for validation
        n = model(xval3)  # Negative output for validation
        
        # Compute validation loss
        valloss = criterion(a, p, n)
        vls += valloss  # Accumulate validation loss
    
    # Print average training and validation loss for the current epoch
    print(f'Epoch: {epoch+1}/{EPOCHS} || Loss: {lss/steps:.4f} || Validation Loss: {vls/vstep:.4f}')


# Load two images from the provided paths and preprocess them
img1 = r"PATH_image1"  # Path to the first image
img2 = r"PATH_image2"  # Path to the second image

# Read, resize, and normalize image1, then convert it to a PyTorch tensor
img1 = torch.from_numpy(np.array(cv2.resize(cv2.imread(img1), (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0, dtype=np.float32))

# Read, resize, and normalize image2, then convert it to a PyTorch tensor
img2 = torch.from_numpy(np.array(cv2.resize(cv2.imread(img2), (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0, dtype=np.float32))

# Reshape the images to the format expected by the model: (batch_size, channels, height, width)
pred1 = model(img1.reshape(1, 3, IMG_SHAPE, IMG_SHAPE))  # Predict features for image1
pred2 = model(img2.reshape(1, 3, IMG_SHAPE, IMG_SHAPE))  # Predict features for image2

# Calculate the pairwise distance between the two predicted feature vectors
distance = torch.pairwise_distance(pred1, pred2)

# Output the calculated distance
print(distance)

# Convert the training list to a NumPy array

