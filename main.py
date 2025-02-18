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

# create dataset class
class BrainTumorDataset(Dataset):


