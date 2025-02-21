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

# create dataset class
#class BrainTumorDataset(Dataset):


training_glioma = []
for i in range(0, 1000):
    val = str(i)
    while(len(val) !=3):
        val = "0"+val
    training_glioma.append("Training/glioma/Tr_gl_0" + val+ ".jpg")
for i in range(0,10):

    img = Image.open(training_glioma[i])

    img.show()
