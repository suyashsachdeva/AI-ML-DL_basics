from utility_functions import preprocessing  # Importing a custom preprocessing function

import torch  # PyTorch library for tensor operations
import torch.utils as utils  # Utility functions for DataLoader and more
import cv2  # OpenCV library for image processing tasks
import numpy as np  # NumPy for numerical operations on arrays
import os  # OS library for file and directory operations
import random  # For generating random values (useful for data augmentations)

from tqdm.auto import trange  # TQDM for creating progress bars in loops

# Path to the dataset folder
PATH = r"dataset_folder_path"

# Defining constants for training configuration
EPOCHS = 10              # Number of training epochs
DATA_SIZE = 5000         # Total size of the training dataset
BATCH_SIZE = 100         # Batch size for training
VALID_DATA = 500         # Total size of the validation dataset
VALID_BATCH = 100        # Batch size for validation
IMG_SHAPE = 64           # Image dimensions (assumes square images of shape 64x64)


def to_torch_and_split(train):
    """
    Convert the input training data into PyTorch tensors and split them into batches for training and validation.

    Args:
        train: A NumPy array representing the training data in the form of triplets (anchor, positive, negative).

    Returns:
        DataLoaders for training and validation data for anchor, positive, and negative images.
    """
    
    # Convert the input 'train' data into a NumPy array of type float32 (for better compatibility with PyTorch)
    x = np.array(train, dtype=np.float32)

    # Convert NumPy arrays to PyTorch tensors for training data (anchor, positive, negative images)
    xtrain1 = torch.from_numpy(x[:DATA_SIZE, 0].reshape(DATA_SIZE, 3, IMG_SHAPE, IMG_SHAPE))  # Anchor images (training)
    xtrain2 = torch.from_numpy(x[:DATA_SIZE, 1].reshape(DATA_SIZE, 3, IMG_SHAPE, IMG_SHAPE))  # Positive images (training)
    xtrain3 = torch.from_numpy(x[:DATA_SIZE, 2].reshape(DATA_SIZE, 3, IMG_SHAPE, IMG_SHAPE))  # Negative images (training)
    # ytrain = torch.from_numpy(y[:DATA_SIZE])  # Uncomment if you have label data for training (optional)

    # Convert NumPy arrays to PyTorch tensors for validation data (anchor, positive, negative images)
    xvalid1 = torch.from_numpy(x[DATA_SIZE:DATA_SIZE + VALID_DATA, 0].reshape(VALID_DATA, 3, IMG_SHAPE, IMG_SHAPE))  # Anchor images (validation)
    xvalid2 = torch.from_numpy(x[DATA_SIZE:DATA_SIZE + VALID_DATA, 1].reshape(VALID_DATA, 3, IMG_SHAPE, IMG_SHAPE))  # Positive images (validation)
    xvalid3 = torch.from_numpy(x[DATA_SIZE:DATA_SIZE + VALID_DATA, 2].reshape(VALID_DATA, 3, IMG_SHAPE, IMG_SHAPE))  # Negative images (validation)
    # yvalid = torch.from_numpy(y[DATA_SIZE:DATA_SIZE + VALID_DATA])  # Uncomment if you have label data for validation (optional)

    # Creating DataLoader objects for training data (anchor, positive, negative)
    xtrain1 = utils.data.DataLoader(xtrain1, batch_size=BATCH_SIZE)  # DataLoader for anchor images (training)
    xtrain2 = utils.data.DataLoader(xtrain2, batch_size=BATCH_SIZE)  # DataLoader for positive images (training)
    xtrain3 = utils.data.DataLoader(xtrain3, batch_size=BATCH_SIZE)  # DataLoader for negative images (training)
    # ytrain = utils.data.DataLoader(ytrain, batch_size=BATCH_SIZE)  # Uncomment if you have labels for training

    # Creating DataLoader objects for validation data (anchor, positive, negative)
    xvalid1 = utils.data.DataLoader(xvalid1, batch_size=VALID_BATCH)  # DataLoader for anchor images (validation)
    xvalid2 = utils.data.DataLoader(xvalid2, batch_size=VALID_BATCH)  # DataLoader for positive images (validation)
    xvalid3 = utils.data.DataLoader(xvalid3, batch_size=VALID_BATCH)  # DataLoader for negative images (validation)
    # yvalid = utils.data.DataLoader(yvalid, batch_size=VALID_BATCH)  # Uncomment if you have labels for validation

    return xtrain1, xtrain2, xtrain3, xvalid1, xvalid2, xvalid3
