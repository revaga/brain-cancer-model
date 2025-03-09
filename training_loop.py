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
    testing_glioma.append("Testing/glioma/Te-gl_" + val+ ".jpg")

    
testing_meningioma = []
for i in range(10, 306): #306 images, skipped first 10
    val = str(i)
    while(len(val) !=4):
        val = "0"+val
    testing_meningioma.append("Testing/meningioma/Te-me_" + val+ ".jpg")

    
testing_notumor = []
for i in range(10,405): #405 images, skipped first 10
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    testing_notumor.append("Testing/notumor/Te-no_" + val+ ".jpg")
    #print("Training/notumor/Tr-no_" + val+ ".jpg")



testing_pituitary = []
for i in range(10,300): #300 images, skipped first 10
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    testing_pituitary.append("Testing/pituitary/Te-pi_" + val+ ".jpg")
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

# convert training images into tensors 
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    #transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
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


# Convert testing images into tensors 
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

class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        image = image.convert("L")  # convert rgb to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

# Create two instances of the class, one with the training data and one with the testing data 
# Use each instance to make a Dataloader using the PyTorch Dataloader class
train_images = traintensor_glioma + traintensor_meningioma + traintensor_notumor + traintensor_pituitary
train_labels = [0] * len(traintensor_glioma) + [1] * len(traintensor_meningioma) + [2] * len(traintensor_notumor) + [3] * len(traintensor_pituitary)
train_dataset = BrainTumorDataset(train_images, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_images = testtensor_glioma + testtensor_meningioma + testtensor_notumor + testtensor_pituitary
test_labels = [0] * len(testtensor_glioma) + [1] * len(testtensor_meningioma) + [2] * len(testtensor_notumor) + [3] * len(testtensor_pituitary)
test_dataset = BrainTumorDataset(test_images, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)



# Create a training loop and a testing loop over the data, using each Dataloader. Each loop should load a batch of data and print it
# Each iteration of the loop should only print the input and output values.
# A loaded batch must consist of matching input-output pairs.
# Create a training loop
# Create a CNN model with flattening and feedforward layers
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1) # first input is 1 instead of 3 because 1 channel for grayscale
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        
        # Flattening layer
        self.flatten = torch.nn.Flatten()
        
        # Feedforward layers
        # 128 * 128 is the size of the flattened output from the last pooling layer
        self.fc1 = torch.nn.Linear(64 * 128 * 128, 128)
        self.fc2 = torch.nn.Linear(128, 4) # 4 classes

    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flattening layer
        x = self.flatten(x)
        
        # Feedforward layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x



# # Training loop
# print("Training Data Batches:")
# for batch_idx, (inputs, labels) in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1}:")
#     print("Inputs:", inputs.shape)  # Print shape of input images
#     print("Labels:", labels)  # Print labels
#     if batch_idx == 2:  # Print first 3 batches for readability
#         break

# # Testing loop
# print("\nTesting Data Batches:")
# for batch_idx, (inputs, labels) in enumerate(test_loader):
#     print(f"Batch {batch_idx + 1}:")
#     print("Inputs:", inputs.shape)  # Print shape of input images
#     print("Labels:", labels) 
#     if batch_idx == 2: 
#         break

# training loop
model = CNNModel()
#  loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train model
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    current_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        # zero gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, labels)
        # backward pass and optimization
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1) #get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = current_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}")
