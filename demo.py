import torch
import torchvision.transforms.v2 as v2
from PIL import Image

#CNN model from our training/testing code
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        # self.bn1 = torch.nn.BatchNorm2d(32)
        # self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        
        # Flattening layer
        self.flatten = torch.nn.Flatten()
        
        # Feedforward layers
        self.fc1 = torch.nn.Linear(64 * 64 * 64, 128)  # Adjusted for flattened output
        self.fc2 = torch.nn.Linear(128, 4)  # 4 classes
        self.dropout = torch.nn.Dropout(p=0.5)  # Added dropout layer

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        #x = self.bn1(x)  # added batch normalization
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        #x = self.bn2(x)  # added batch normalization
        x = self.relu(x)
        x = self.pool(x)
        
        # Flattening layer
        x = self.flatten(x)
        
        # Feedforward layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Added dropout
        x = self.fc2(x)
        
        return x

model = CNNModel()


model_path = 'brain_tumor_cnn.pth' #file path of our model
model.load_state_dict(torch.load(model_path))

model.eval() #evaluation mode, ref: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch

imgpaths = [["glioma", "Training/glioma/Tr-glTr_0007.jpg"], ["meningioma", "Training/meningioma/Tr-meTr_0003.jpg"], ["notumor", "Training/notumor/Tr-noTr_0008.jpg"], ["pituitary", "Training/pituitary/Tr-piTr_0001.jpg"]]
for i in imgpaths:
    image_path = i[1]
    image = Image.open(image_path).convert("L") 
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(size=(256, 256), antialias=True),
        v2.ConvertImageDtype(torch.float32),
    ])

    with torch.no_grad(): #don't update model based on output
        image_tensor = transform(image).unsqueeze(0)  #unsqueeze because dimension issues
        output = model(image_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0) #ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
    #https://www.geeksforgeeks.org/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/?form=MG0AV3
  
    predicted_class = torch.argmax(probabilities).item()
    print(f"Actual class: {i[0]}")
  
    #matching labels with model to output in reader-friendly manner
    if(predicted_class == 0):
        print(f"Predicted class: glioma")
    elif(predicted_class == 1):
        print(f"Predicted class: meningioma")
    elif(predicted_class == 2):
        print(f"Predicted class: notumor")
    elif(predicted_class == 3):
        print(f"Predicted class: pituitary")
    print(f"Probabilities: {probabilities}")
    print()
