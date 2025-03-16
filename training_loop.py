# creating CNN to predict brain cancer

# importing libraries
# importing libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image 
import torchvision.transforms as transforms
import torchvision
import torchvision.transforms.v2 as v2
import wandb
from torch.optim.lr_scheduler import ExponentialLR as Explr


# create dataset class
#class BrainTumorDataset(Dataset):

alltrainimgs = []


training_glioma = []
for i in range(10, 1321): #skipped first 10
    value = str(i)
    while(len(value) !=4):
        value = "0"+value
    training_glioma.append("Training/glioma/Tr-gl_" + value+ ".jpg")
    alltrainimgs.append("Training/glioma/Tr-gl_" + value+ ".jpg")

    
training_meningioma = []
for i in range(10, 1339): #1338 images, skipped first 10
    value = str(i)
    while(len(value) !=4):
        value = "0"+value
    training_meningioma.append("Training/meningioma/Tr-me_" + value+ ".jpg")
    alltrainimgs.append("Training/meningioma/Tr-me_" + value+ ".jpg")

    
training_notumor = []
for i in range(10,1595): #1594 images, first 10 ommitted
    value = str(i)
    while(len(value) != 4):
        value = "0" + value
    training_notumor.append("Training/notumor/Tr-no_" + value+ ".jpg")
    alltrainimgs.append("Training/notumor/Tr-no_" + value+ ".jpg")
    #print("Training/notumor/Tr-no_" + val+ ".jpg")



training_pituitary = []
for i in range(10,1457): #1456 images, first 10 ommitted
    value = str(i)
    while(len(value) != 4):
        value = "0" + value
    training_pituitary.append("Training/pituitary/Tr-pi_" + value+ ".jpg")
    alltrainimgs.append("Training/pituitary/Tr-pi_" + value+ ".jpg")
    #print(training_pituitary[i])



alltestimgs = []
allvalimgs = []

allvallengths = [0,0,0,0]

testing_glioma = []
for i in range(10, 300): #300 images, skipped first 10
    value = str(i)
    while(len(value) !=4):
        value = "0"+value
    if(i < 150):
        testing_glioma.append("Testing/glioma/Te-gl_" + value+ ".jpg")
        alltestimgs.append("Testing/glioma/Te-gl_" + value+ ".jpg")
    else:
        allvalimgs.append("Testing/glioma/Te-gl_" + value+ ".jpg")
        allvallengths[0] = allvallengths[0] + 1


    
testing_meningioma = []
for i in range(10, 306): #306 images, skipped first 10
    value = str(i)
    while(len(value) !=4):
        value = "0"+value
    if(i < 153):
        testing_meningioma.append("Testing/meningioma/Te-me_" + value+ ".jpg")
        alltestimgs.append("Testing/meningioma/Te-me_" + value+ ".jpg")
    else:
        allvalimgs.append("Testing/meningioma/Te-me_" + value+ ".jpg")
        allvallengths[1] = allvallengths[1] + 1


    
testing_notumor = []
for i in range(10,405): #405 images, skipped first 10
    value = str(i)
    while(len(value) != 4):
        value = "0" + value
    if(i < 202):
        testing_notumor.append("Testing/notumor/Te-no_" + value+ ".jpg")
        alltestimgs.append("Testing/notumor/Te-no_" + value+ ".jpg")
    else:
        allvalimgs.append("Testing/notumor/Te-no_" + value+ ".jpg")
        allvallengths[2] = allvallengths[2] + 1

    #print("Training/notumor/Tr-no_" + val+ ".jpg")



testing_pituitary = []
for i in range(10,300): #300 images, skipped first 10
    value = str(i)
    while(len(value) != 4):
        value = "0" + value
    if (i < 150):
        testing_pituitary.append("Testing/pituitary/Te-pi_" + value+ ".jpg")
        alltestimgs.append("Testing/pituitary/Te-pi_" + value+ ".jpg")
    else:
        allvalimgs.append("Testing/pituitary/Te-pi_" + value+ ".jpg")
        allvallengths[3] = allvallengths[3] + 1

    #print(testing_pituitary[i])

#ref https://pytorch.org/vision/master/auto_examples/transforms/plot_transforms_getting_started.html
transform = v2.Compose([ 
    v2.PILToTensor(),
    v2.Resize(size=(256, 256)),
    v2.ConvertImageDtype(torch.float32), #convertDtype, toDtype
    v2.Grayscale(num_output_channels=1),
    v2.RandomHorizontalFlip(p=0.5), 
    v2.RandomRotation(degrees=15), 
    v2.RandomResizedCrop(size=(256,256), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
])

class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        #image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            

        #image = image.convert("L")  # convert rgb to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
    


#traintensor_glioma = []
"""
traintensor_glioma = torch.load('traintensor_glioma.pt')
traintensor_meningioma = torch.load('traintensor_meningioma.pt')
traintensor_pituitary = torch.load('traintensor_pituitary.pt')
traintensor_notumor = torch.load('traintensor_notumor.pt')

testtensor_glioma = torch.load('testtensor_glioma.pt')
testtensor_meningioma = torch.load('testtensor_meningioma.pt')
testtensor_pituitary = torch.load('testtensor_pituitary.pt')
testtensor_notumor = torch.load('testtensor_glioma.pt')
"""





# Create two instances of the class, one with the training data and one with the testing data 
# Use each instance to make a Dataloader using the PyTorch Dataloader class
#train_images = traintensor_glioma + traintensor_meningioma + traintensor_notumor + traintensor_pituitary
train_images = alltrainimgs
#print(len(alltrainimgs))
#print(len(training_glioma) + len(training_meningioma) + len(training_notumor) + len(training_pituitary))
train_labels = [0] * len(training_glioma) + [1] * len(training_meningioma) + [2] * len(training_notumor) + [3] * len(training_pituitary)
train_dataset = BrainTumorDataset(train_images, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


test_images = alltestimgs
test_labels = [0] * len(testing_glioma) + [1] * len(testing_meningioma) + [2] * len(testing_notumor) + [3] * len(testing_pituitary)
test_dataset = BrainTumorDataset(test_images, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


val_imags = allvalimgs
val_labels = [0] * allvallengths[0] + [1] * allvallengths[1] + [2] * allvallengths[2] + [3] * allvallengths[3]
print(len(val_imags))
print(len(val_labels))
val_dataset = BrainTumorDataset(val_imags, val_labels, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# Create a training loop and a testing loop over the data, using each Dataloader. Each loop should load a batch of data and print it
# Each iteration of the loop should only print the input and output values.
# A loaded batch must consist of matching input-output pairs.
# Create a training loop
# Create a CNN model with flattening and feedforward layers
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel for grayscale

        #self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1) # first input is 1 instead of 3 because 1 channel for grayscale
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        
        # Flattening layer
        self.flatten = torch.nn.Flatten()
        
        # Feedforward layers
        # 128 * 128 is the size of the flattened output from the last pooling layer
        self.fc1 = torch.nn.Linear(64 * 64 * 64, 128)
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



#WandB

run = wandb.init(project="CMPM17-BCM", name="run-thursday3")

# training loop
model = CNNModel()
#  loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = Explr(optimizer, gamma=0.9)
# train model
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    train_current_loss = 0
    train_correct = 0
    train_total = 0


    val_current_loss = 0
    val_correct = 0
    val_total = 0


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
        train_current_loss += loss.item()
        run.log({"train/smallloss": loss.item()})
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        _, predicted = torch.max(outputs.data, 1) # gives the index of the max log-probability - the predicted class
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        scheduler.step()
    epoch_loss = train_current_loss / len(train_loader)
    epoch_accuracy = train_correct / train_total        
    run.log({"train/totalloss": train_current_loss})
    run.log({"train/accuracy": epoch_accuracy})
    run.log({"train/epochloss": epoch_loss})

    model.eval()
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            #validation
            val_pred = model(val_inputs)
            val_loss = criterion(val_pred, val_labels)
            run.log({"val/smallloss": val_loss.item()})
            val_current_loss += val_loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], VAL Loss: {val_loss.item():.4f}")
            _, val_predicted = torch.max(val_pred.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
        run.log({"val/accuracy": val_correct / val_total})
        run.log({"val/totalloss": val_current_loss})
        run.log({"val/epochloss": val_current_loss / len(val_loader)})


#testing loop
test_current_loss = 0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # forward pass
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, labels)

        test_current_loss += loss.item()
        run.log({"testsmallloss": loss.item()})
        #print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        _, predicted = torch.max(outputs.data, 1) #get the index of the max log-probability
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
    epoch_loss = test_current_loss / len(test_loader)
    epoch_accuracy = test_correct / test_total
    run.log({"testtotalloss": test_current_loss})
    run.log({"testaccuracy": epoch_accuracy})
