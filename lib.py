# download data
import os
import os.path as osp
import tarfile
import zipfile
import urllib.request

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import functional as F
import glob
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms
import torch.utils.data as data


from PIL import Image
import matplotlib.pyplot as plt