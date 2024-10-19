import cv2  # OpenCV for image processing tasks
import numpy as np  # NumPy for numerical operations
import os  # For file and directory operations
import random  # For generating random values

from tqdm.auto import trange  # TQDM for progress bars in loops


# Defining constants for training configuration
EPOCHS = 10              # Number of training epochs
DATA_SIZE = 5000         # Total size of the training dataset
BATCH_SIZE = 100         # Batch size for training
VALID_DATA = 500         # Total size of the validation dataset
VALID_BATCH = 100        # Batch size for validation
IMG_SHAPE = 64        

def preprocessing(FOLDER):
    folders = os.listdir(FOLDER)  # List all subfolders (classes)
    train = []  # List to store triplet (anchor, positive, negative) samples

    for folder in folders:
        files = os.listdir(FOLDER + "/" + folder)  # List all files in the current folder

        for i in range(len(files) - 1):
            nfolder = random.choice(folders)  # Randomly select a different folder for the negative sample

            if nfolder != folder:
                path = r"PATH"  # Path for negative sample from another folder
                negative = cv2.imread(path)  # Load negative image from a different folder
            else:
                path = r"PATH"  # Path for negative sample (you may want to change this logic)
                negative = cv2.imread(path)  # Load negative image from the same folder

            # Load the anchor and positive samples from the current folder
            anchor = cv2.imread(FOLDER + "/" + folder + "/" + files[i])
            positive = cv2.imread(FOLDER + "/" + folder + "/" + files[i + 1])

            # Resize the images to the desired shape and normalize
            anchor = cv2.resize(anchor, (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0
            positive = cv2.resize(positive, (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0
            negative = cv2.resize(negative, (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0

            # Append the triplet (anchor, positive, negative) to the training list
            train.append([anchor, positive, negative])

        # For folders with 4 or more files, create additional triplet samples
        if len(files) >= 4:
            for i in range(2):
                if nfolder != folder:
                    fk = os.listdir(FOLDER + "/" + nfolder)[0]  # Select the first file from the negative folder
                    negative = cv2.imread(FOLDER + "/" + nfolder + "/" + fk)
                else:
                    path = r"PATH"  # Path for the negative sample
                    negative = cv2.imread(path)

                # Load the anchor and positive samples from the current folder
                anchor = cv2.imread(FOLDER + "/" + folder + "/" + files[-2 + i])
                positive = cv2.imread(FOLDER + "/" + folder + "/" + files[0 + i])

                # Resize and normalize the images
                anchor = cv2.resize(anchor, (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0
                positive = cv2.resize(positive, (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0
                negative = cv2.resize(negative, (IMG_SHAPE, IMG_SHAPE), cv2.INTER_LINEAR) / 255.0

                # Append the triplet to the training list
                train.append([anchor, positive, negative])

    return train