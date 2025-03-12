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
import torchvision.transforms.v2 as v2
import random

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


# Convert training images into tensors 
# ref: https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
"""transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1000,1000)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(degrees=15), 
    transforms.RandomResizedCrop(size=(512,512), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    ])
    """



transform2 = v2.Compose([ 
    v2.PILToTensor(),
    v2.Resize(size=(256, 256)),
    v2.ConvertImageDtype(torch.float32), #convertDtype, toDtype
    #v2.Grayscale(num_output_channels=1),
    v2.RandomHorizontalFlip(p=0.5), 
    v2.RandomRotation(degrees=50), 
    v2.RandomResizedCrop(size=(512,512), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
])

"""
for i in range(len(training_glioma)):
    image = Image.open(training_glioma[i])
    newimg = transform(image)
    newimg.save(f"Training/glioma/aug/Tr-gl-aug_{i}.jpg")

for i in range(len(training_meningioma)):
    image = Image.open(training_meningioma[i])
    newimg = transform(image)
    newimg.save(f"Training/meningioma/aug/Tr-me-aug_{i}.jpg")
"""
img = Image.open(training_notumor[3])
print(img.mode)  # Should now print 'RGB'
img.show()
print(f"Pre Image Shape: {img.size}")

for i in range(len(training_notumor)):
    image = Image.open(training_notumor[i]).convert("L")
    newimg = transform2(image)
    
    #newimg.save(f"Training/notumor/aug/Tr-no-aug_{i}.png")

img = Image.open(training_notumor[3]) # Convert to grayscale immediately
print(img.mode)  # Should now print 'L'
print(f"Transformed Image Shape: {img.size}")
img.show()
    
"""
for i in range(len(training_pituitary)):
    image = Image.open(training_pituitary[i])
    newimg = transform(image)
"""
    #newimg.save(f"Training/pituitary/aug/Tr-pi-aug_{i}.jpg")

