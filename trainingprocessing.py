# creating CNN to predict brain cancer

# importing libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os

from PIL import Image 
import torchvision.transforms as transforms

# create dataset class
#class BrainTumorDataset(Dataset):


training_glioma = []
for i in range(10, 1321): #skipped first 10
    val = str(i)
    while(len(val) !=4):
        val = "0"+val
    training_glioma.append("Training/glioma/Tr-gl_" + val+ ".jpg")

    
training_meningioma = []
for i in range(10, 1339): #1338 images, skipped first 10
    val = str(i)
    while(len(val) !=4):
        val = "0"+val
    training_meningioma.append("Training/meningioma/Tr-me_" + val+ ".jpg")

    
training_notumor = []
for i in range(10,1595): #1594 images, first 10 ommitted
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    training_notumor.append("Training/notumor/Tr-no_" + val+ ".jpg")
    #print("Training/notumor/Tr-no_" + val+ ".jpg")



training_pituitary = []
for i in range(10,1457): #1456 images, first 10 ommitted
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    training_pituitary.append("Training/pituitary/Tr-pi_" + val+ ".jpg")
    #print(training_pituitary[i])


"""
#showing images

for i in range(0,2):
    img = Image.open(training_glioma[i])
    img.show()


for i in range(0,2):
    img = Image.open(training_meningioma[i])
    img.show()

for i in range(0,2):
    img = Image.open(training_notumor[i])
    img.show()

for i in range(0,2):
    img = Image.open(training_pituitary[i])
    img.show()

"""

# Convert training images into tensors 
# ref: https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
transform = transforms.Compose([
    transforms.PILToTensor()
])

traintensor_glioma = []
for i in range(0, len(training_glioma)):
    img = Image.open(training_glioma[i])
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_glioma.append(img_tensor)

traintensor_meningioma = []
for i in range(0, len(training_meningioma)):
    img = Image.open(training_meningioma[i])
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_meningioma.append(img_tensor)

traintensor_notumor = []
for i in range(0, len(training_notumor)):
    img = Image.open(training_notumor[i])
    img_tensor = transform(img)
    traintensor_notumor.append(img_tensor)

traintensor_pituitary = []
for i in range(0, len(training_pituitary)):
    img = Image.open(training_pituitary[i])
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_pituitary.append(img_tensor)



