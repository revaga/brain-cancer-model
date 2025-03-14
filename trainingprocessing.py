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


testing_glioma = []
for i in range(10, 300): #300 images, skipped first 10
    val = str(i)
    while(len(val) !=4):
        val = "0"+val
    testing_glioma.append("Training/glioma/Tr-gl_" + val+ ".jpg")

    
testing_meningioma = []
for i in range(10, 306): #306 images, skipped first 10
    val = str(i)
    while(len(val) !=4):
        val = "0"+val
    testing_meningioma.append("Training/meningioma/Tr-me_" + val+ ".jpg")

    
testing_notumor = []
for i in range(10,405): #405 images, skipped first 10
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    testing_notumor.append("Training/notumor/Tr-no_" + val+ ".jpg")
    #print("Training/notumor/Tr-no_" + val+ ".jpg")



testing_pituitary = []
for i in range(10,300): #300 images, skipped first 10
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    testing_pituitary.append("Training/pituitary/Tr-pi_" + val+ ".jpg")
    #print(testing_pituitary[i])

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


testtensor_glioma = []
for i in range(0, len(testing_glioma)):
    img = Image.open(testing_glioma[i])
    img_tensor = transform(img)
    #print(img_tensor)
    testtensor_glioma.append(img_tensor)

testtensor_meningioma = []
for i in range(0, len(testing_meningioma)):
    img = Image.open(testing_meningioma[i])
    img_tensor = transform(img)
    #print(img_tensor)
    testtensor_meningioma.append(img_tensor)

testtensor_notumor = []
for i in range(0, len(testing_notumor)):
    img = Image.open(testing_notumor[i])
    img_tensor = transform(img)
    testtensor_notumor.append(img_tensor)

testtensor_pituitary = []
for i in range(0, len(testing_pituitary)):
    img = Image.open(testing_pituitary[i])
    img_tensor = transform(img)
    #print(img_tensor)
    testtensor_pituitary.append(img_tensor)



"""

traintensor_glioma = []
for i in range(0, len(training_glioma)):
    img = Image.open(training_glioma[i]).convert("L")
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_glioma.append(img_tensor)

#https://pytorch.org/docs/stable/generated/torch.save.html
torch.save(traintensor_glioma, "traintensor_glioma.pt")

"""
img_tensor = traintensor_glioma[0]
print(img_tensor.shape) #should be 256x256

#ref: https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ToPILImage.html
img_pil = v2.ToPILImage()(img_tensor)
print(img_pil.mode)  # should be 'L' for grayscale
img_pil.show()
"""

traintensor_meningioma = []
for i in range(0, len(training_meningioma)):
    img = Image.open(training_meningioma[i]).convert("L")
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_meningioma.append(img_tensor)
torch.save(traintensor_meningioma, "traintensor_meningioma.pt")


traintensor_notumor = []
for i in range(0, len(training_notumor)):
    img = Image.open(training_notumor[i]).convert("L")
    img_tensor = transform(img)
    traintensor_notumor.append(img_tensor)
torch.save(traintensor_notumor, "traintensor_notumor.pt")



traintensor_pituitary = []
for i in range(0, len(training_pituitary)):
    img = Image.open(training_pituitary[i]).convert("L")
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_pituitary.append(img_tensor)
torch.save(traintensor_pituitary, "traintensor_pituitary.pt")


# Convert testing images into tensors 
testtensor_glioma = []
for i in range(0, len(testing_glioma)):
    img = Image.open(testing_glioma[i]).convert("L")
    img_tensor = transform(img)
    #print(img_tensor)
    testtensor_glioma.append(img_tensor)
torch.save(testtensor_glioma, "testtensor_glioma.pt")


testtensor_meningioma = []
for i in range(0, len(testing_meningioma)):
    img = Image.open(testing_meningioma[i]).convert("L")
    img_tensor = transform(img)
    #print(img_tensor)
    testtensor_meningioma.append(img_tensor)
torch.save(testtensor_meningioma, "testtensor_meningioma.pt")


testtensor_notumor = []
for i in range(0, len(testing_notumor)):
    img = Image.open(testing_notumor[i]).convert("L")
    img_tensor = transform(img)
    testtensor_notumor.append(img_tensor)
torch.save(testtensor_notumor, "testtensor_notumor.pt")

testtensor_pituitary = []
for i in range(0, len(testing_pituitary)):
    img = Image.open(testing_pituitary[i]).convert("L")
    img_tensor = transform(img)
    #print(img_tensor)
    testtensor_pituitary.append(img_tensor)
torch.save(testtensor_pituitary, "testtensor_pituitary.pt")
"""