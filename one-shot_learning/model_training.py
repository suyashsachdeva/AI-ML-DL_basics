import torch  # PyTorch for building and training models
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import numpy as np  # NumPy for numerical operations
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import DataLoader, Dataset  # Dataloader and dataset management
from tqdm import trange  # Progress bar for loops

import cv2  # OpenCV for image processing
from loss_function import Loss  # Custom loss function for the model
from complete_model import OneShot  # Import the OneShot model architecture
from utility_functions import preprocessing  # Preprocessing utility functions
from data_preprocessing import to_torch_and_split  # Function to preprocess data and split into PyTorch tensors

# Set path to the dataset folder
PATH = r"folder"

# Define constants for training configuration
EPOCHS = 10              # Number of training epochs
DATA_SIZE = 5000         # Total size of the training dataset
BATCH_SIZE = 100         # Batch size for training
VALID_DATA = 500         # Total size of the validation dataset
VALID_BATCH = 100        # Batch size for validation
IMG_SHAPE = 64           # Image dimensions (assuming square images of 64x64 pixels)

# Preprocess the dataset using custom preprocessing function
train = preprocessing(PATH)
# Convert the preprocessed data into PyTorch tensors and split it into training and validation sets
xtrain1, xtrain2, xtrain3, xvalid1, xvalid2, xvalid3 = to_torch_and_split(train)

# Calculate the number of steps per epoch for training and validation
steps = len(xtrain1)  # Number of steps (batches) in training data
vstep = len(xvalid1)  # Number of steps (batches) in validation data

# Initialize the OneShot model
model = OneShot()

# Define the triplet loss function for calculating the pairwise distance
criterion = Loss()

# Set the initial learning rate and decay factor for learning rate scheduling
learning_rate = 1e-4
decay = 0.9  # Learning rate decay over time

# Training loop over the number of epochs
for epoch in trange(EPOCHS):
    training_loss = 0  # Initialize the total training loss for the epoch
    validation_loss = 0  # Initialize the total validation loss for the epoch
    
    # Decay the learning rate with each epoch
    learning_rate = learning_rate / (1 + epoch * decay)
    
    # Initialize the optimizer with the updated learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop over batches
    for step, (xtr1, xtr2, xtr3) in enumerate(zip(xtrain1, xtrain2, xtrain3)):
        anchor = model(xtr1)  # Anchor output
        positive = model(xtr2)  # Positive output
        negative = model(xtr3)  # Negative output
        
        # Compute the triplet loss
        loss = criterion(anchor, positive, negative)
        training_loss += loss.item()  # Accumulate training loss
        
        # Zero the gradients, backpropagate, and update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop over batches
    for step, (xval1, xval2, xval3) in enumerate(zip(xvalid1, xvalid2, xvalid3)):
        anchor = model(xval1)  # Anchor output for validation
        positive = model(xval2)  # Positive output for validation
        negative = model(xval3)  # Negative output for validation
        
        # Compute the validation loss
        val_loss = criterion(anchor, positive, negative)
        validation_loss += val_loss.item()  # Accumulate validation loss
    
    # Print the average training and validation loss for the current epoch
    print(f'Epoch: {epoch+1}/{EPOCHS} || Loss: {training_loss/steps:.4f} || Validation Loss: {validation_loss/vstep:.4f}')

# Load two images for comparison
img1_path = r"PATH_image1"  # Path to the first image
img2_path = r"PATH_image2"  # Path to the second image

# Read, resize, and normalize the first image, then convert it to a PyTorch tensor
img1 = torch.from_numpy(np.array(cv2.resize(cv2.imread(img1_path), (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0, dtype=np.float32))

# Read, resize, and normalize the second image, then convert it to a PyTorch tensor
img2 = torch.from_numpy(np.array(cv2.resize(cv2.imread(img2_path), (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0, dtype=np.float32))

# Reshape the images to the format expected by the model: (batch_size, channels, height, width)
pred1 = model(img1.reshape(1, 3, IMG_SHAPE, IMG_SHAPE))  # Predict features for image1
pred2 = model(img2.reshape(1, 3, IMG_SHAPE, IMG_SHAPE))  # Predict features for image2

# Calculate the pairwise distance between the two predicted feature vectors
distance = torch.pairwise_distance(pred1, pred2)

# Output the calculated distance
print(f"Distance between images: {distance.item():.4f}")

# Convert the training list to a NumPy array
# (Uncomment or modify depending on how the training data is handled)
# train_np_array = np.array(train)
