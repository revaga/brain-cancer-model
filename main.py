# creating CNN to predict brain cancer

# importing libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import torchvision.transforms as transforms
import torchvision
import torchvision.transforms.v2 as v2
import wandb

#checking if GPU is available, otherwise running on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#array with file paths for all the training images
train_images = []

#file paths for glioma training images
training_glioma = []
for i in range(10, 1321):  # 1311 images, skipped first 10
    value = str(i)
    while len(value) != 4:
        value = "0" + value
    training_glioma.append("Training/glioma/Tr-gl_" + value + ".jpg")
    train_images.append("Training/glioma/Tr-gl_" + value + ".jpg")


#file paths for meningioma training images
training_meningioma = []
for i in range(10, 1339):  # 1338 images, skipped first 10
    value = str(i)
    while len(value) != 4:
        value = "0" + value
    training_meningioma.append("Training/meningioma/Tr-me_" + value + ".jpg")
    train_images.append("Training/meningioma/Tr-me_" + value + ".jpg")


#file paths for no tumor training images
training_notumor = []
for i in range(10, 1595):  # 1594 images, first 10 omitted
    value = str(i)
    while len(value) != 4:
        value = "0" + value
    training_notumor.append("Training/notumor/Tr-no_" + value + ".jpg")
    train_images.append("Training/notumor/Tr-no_" + value + ".jpg")


#file paths for pituitary training images
training_pituitary = []
for i in range(10, 1457):  # 1456 images, first 10 omitted
    value = str(i)
    while len(value) != 4:
        value = "0" + value
    training_pituitary.append("Training/pituitary/Tr-pi_" + value + ".jpg")
    train_images.append("Training/pituitary/Tr-pi_" + value + ".jpg")

#file paths for all testing images
test_images = []

#file paths for all validation images
val_imags = []

#keeping track of val lengths for each class to be used in creating labels in dataloader
allvallengths = [0, 0, 0, 0]

#file paths for testing glioma images
testing_glioma = []
for i in range(10, 300):  # 300 images, skipped first 10
    value = str(i)
    while len(value) != 4:
        value = "0" + value

    #separate half of images into validation
    if i < 150:
        testing_glioma.append("Testing/glioma/Te-gl_" + value + ".jpg")
        test_images.append("Testing/glioma/Te-gl_" + value + ".jpg")
    else:
        val_imags.append("Testing/glioma/Te-gl_" + value + ".jpg")
        allvallengths[0] += 1

#file paths for testing glioma images
testing_meningioma = []
for i in range(10, 306):  # 306 images, skipped first 10
    value = str(i)
    while len(value) != 4:
        value = "0" + value
    
    #separate half of images into validation
    if i < 153:
        testing_meningioma.append("Testing/meningioma/Te-me_" + value + ".jpg")
        test_images.append("Testing/meningioma/Te-me_" + value + ".jpg")
    else:
        val_imags.append("Testing/meningioma/Te-me_" + value + ".jpg")
        allvallengths[1] += 1

#file paths for no tumor testing images
testing_notumor = []
for i in range(10, 405):  # 405 images, skipped first 10
    value = str(i)
    while len(value) != 4:
        value = "0" + value

    #separate half for validation
    if i < 202:
        testing_notumor.append("Testing/notumor/Te-no_" + value + ".jpg")
        test_images.append("Testing/notumor/Te-no_" + value + ".jpg")
    else:
        val_imags.append("Testing/notumor/Te-no_" + value + ".jpg")
        allvallengths[2] += 1

#file paths for pituitary testing images
testing_pituitary = []
for i in range(10, 300):  # 300 images, skipped first 10
    value = str(i)
    while len(value) != 4:
        value = "0" + value

    #separate half for validation
    if i < 150:
        testing_pituitary.append("Testing/pituitary/Te-pi_" + value + ".jpg")
        test_images.append("Testing/pituitary/Te-pi_" + value + ".jpg")
    else:
        val_imags.append("Testing/pituitary/Te-pi_" + value + ".jpg")
        allvallengths[3] += 1


# Transformations for training
transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(size=(256, 256), antialias=True), #to keep image size consistent
    v2.ConvertImageDtype(torch.float32), #to keep tensor types consistent, ref: https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ConvertImageDtype.html
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
])

# Transformations for validation and testing, no data augmentations except resize(for consistency)
val_transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(size=(256, 256), antialias=True),
    v2.ConvertImageDtype(torch.float32),
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
        label = self.labels[idx]

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# Create instances of the dataset and dataloaders
#creating labels, batching data, and creating dataloader for training data
train_labels = [0] * len(training_glioma) + [1] * len(training_meningioma) + [2] * len(training_notumor) + [3] * len(training_pituitary)
train_dataset = BrainTumorDataset(train_images, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)

#creating labels, batching data, and creating dataloader for testing data
test_labels = [0] * len(testing_glioma) + [1] * len(testing_meningioma) + [2] * len(testing_notumor) + [3] * len(testing_pituitary)
test_dataset = BrainTumorDataset(test_images, test_labels, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

#creating labels, batching data, and creating dataloader for validation data
val_labels = [0] * allvallengths[0] + [1] * allvallengths[1] + [2] * allvallengths[2] + [3] * allvallengths[3]
val_dataset = BrainTumorDataset(val_imags, val_labels, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

# Create a CNN model with flattening and feedforward layers
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)  # First convolutional layer, maintains size
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        # self.bn1 = torch.nn.BatchNorm2d(32)
        # self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        
        # Flattening layer
        self.flatten = torch.nn.Flatten()
        
        # Feedforward layers
        self.fc1 = torch.nn.Linear(64 * 64 * 64, 128)  # Adjusted for flattened output
        self.fc2 = torch.nn.Linear(128, 4)  # 4 classes (glioma: 0, meningioma: 1, no tumor: 2, pituitary: 3)
        self.dropout = torch.nn.Dropout(p=0.5)  # Added dropout layer

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flattening layer
        x = self.flatten(x)
        
        # Feedforward layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  #dropout to improve model's learning
        x = self.fc2(x)
        
        return x

# WandB initialization
run = wandb.init(project="CMPM17-BCM", name="run")

# Training loop
model = CNNModel()

# Move model to GPU if available
model.to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 30
for epoch in range(num_epochs):
    model.train() #training mode
    train_current_loss = 0
    train_correct = 0
    train_total = 0

    val_current_loss = 0
    val_correct = 0
    val_total = 0

    for inputs, labels in train_loader: 
        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_current_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)  # Predicted class
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    epoch_loss = train_current_loss / len(train_loader)
    epoch_accuracy = train_correct / train_total        
    
    run.log({"train/accuracy": epoch_accuracy})
    run.log({"train/epochloss": epoch_loss})


    model.eval() #evaluation mode for validation
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            # Move data to GPU
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            # Validation
            val_pred = model(val_inputs)
            val_loss = criterion(val_pred, val_labels)
            val_current_loss += val_loss.item()
            
            _, val_predicted = torch.max(val_pred.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
        
        run.log({"val/accuracy": val_correct / val_total})
        run.log({"val/epochloss": val_current_loss / len(val_loader)})

torch.save(model.state_dict(), "brain_tumor_cnn.pth")
print("model saved")

#load trained model before testing
model = CNNModel().to(device)
model.load_state_dict(torch.load("brain_tumor_cnn.pth"))

model.eval()  # evaluation mode
print("model loaded")

# Testing loop
test_current_loss = 0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, labels)
        test_current_loss += loss.item()        
        _, predicted = torch.max(outputs.data, 1)  # Predicted class
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
    
    epoch_loss = test_current_loss / len(test_loader)
    epoch_accuracy = test_correct / test_total
    
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_accuracy:.4f}")
    run.log({"testtotalloss": test_current_loss})
    run.log({"testaccuracy": epoch_accuracy})
