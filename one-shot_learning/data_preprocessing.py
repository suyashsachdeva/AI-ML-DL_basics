from utility_functions import preprocessing

import torch 
import torch.utils as utils
import cv2  # OpenCV for image processing tasks
import numpy as np  # NumPy for numerical operations
import os  # For file and directory operations
import random  # For generating random values

from tqdm.auto import trange  # TQDM for progress bars in loops

PATH = r"dataset_folder_path"
# Defining constants for training configuration
EPOCHS = 10              # Number of training epochs
DATA_SIZE = 5000         # Total size of the training dataset
BATCH_SIZE = 100         # Batch size for training
VALID_DATA = 500         # Total size of the validation dataset
VALID_BATCH = 100        # Batch size for validation
IMG_SHAPE = 64           # Image dimensions (assumes square images of shape 64x64)



def to_torch_and_split(train)
    x = np.array(train, dtype=np.float32)


    # Convert NumPy arrays to PyTorch tensors for the training data (anchor, positive, negative)
    xtrain1 = torch.from_numpy(x[:DATA_SIZE, 0].reshape(DATA_SIZE, 3, IMG_SHAPE, IMG_SHAPE))  # Anchor images
    xtrain2 = torch.from_numpy(x[:DATA_SIZE, 1].reshape(DATA_SIZE, 3, IMG_SHAPE, IMG_SHAPE))  # Positive images
    xtrain3 = torch.from_numpy(x[:DATA_SIZE, 2].reshape(DATA_SIZE, 3, IMG_SHAPE, IMG_SHAPE))  # Negative images
    # ytrain = torch.from_numpy(y[:DATA_SIZE])  # Uncomment if you have labels (optional)

    # Convert NumPy arrays to PyTorch tensors for the validation data (anchor, positive, negative)
    xvalid1 = torch.from_numpy(x[DATA_SIZE:DATA_SIZE + VALID_DATA, 0].reshape(VALID_DATA, 3, IMG_SHAPE, IMG_SHAPE))  # Anchor images
    xvalid2 = torch.from_numpy(x[DATA_SIZE:DATA_SIZE + VALID_DATA, 1].reshape(VALID_DATA, 3, IMG_SHAPE, IMG_SHAPE))  # Positive images
    xvalid3 = torch.from_numpy(x[DATA_SIZE:DATA_SIZE + VALID_DATA, 2].reshape(VALID_DATA, 3, IMG_SHAPE, IMG_SHAPE))  # Negative images
    # yvalid = torch.from_numpy(y[DATA_SIZE:DATA_SIZE + VALID_DATA])  # Uncomment if you have labels (optional)


    # Creating DataLoader objects for training data (anchor, positive, negative)
    xtrain1 = utils.data.DataLoader(xtrain1, batch_size=BATCH_SIZE)  # DataLoader for anchor images in training
    xtrain2 = utils.data.DataLoader(xtrain2, batch_size=BATCH_SIZE)  # DataLoader for positive images in training
    xtrain3 = utils.data.DataLoader(xtrain3, batch_size=BATCH_SIZE)  # DataLoader for negative images in training
    # ytrain = utils.data.DataLoader(ytrain, batch_size=BATCH_SIZE)  # Uncomment if you have labels

    # Creating DataLoader objects for validation data (anchor, positive, negative)
    xvalid1 = utils.data.DataLoader(xvalid1, batch_size=VALID_BATCH)  # DataLoader for anchor images in validation
    xvalid2 = utils.data.DataLoader(xvalid2, batch_size=VALID_BATCH)  # DataLoader for positive images in validation
    xvalid3 = utils.data.DataLoader(xvalid3, batch_size=VALID_BATCH)  # DataLoader for negative images in validation
    # yvalid = utils.data.DataLoader(yvalid, batch_size=VALID_BATCH)  # Uncomment if you have labels

    return xtrain1, xtrain2, xtrain3, xvalid1, xvalid2, xvalid3


