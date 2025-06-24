import numpy as np
from datetime import datetime
from prettytable import PrettyTable
import csv
import os

def preprocess_volume(volume):
    volume = volume.astype(np.float16)
    low = np.percentile(volume, 10, axis=(0, 1, 2))
    high = np.percentile(volume, 99, axis=(0, 1, 2))
    volume = (volume - low) / (high - low)
    return np.clip(volume, 0, 1).astype(np.float16)

def preprocess_mask(mask):
    #mask = np.invert(mask)
    return np.clip(mask, 0, 1)

def save_data(*args):
    filename = 'data_file.csv'
    arr = []
    for s in args:
        arr.append(s)
    
    # if exists(filename):
    #     prefix, ext = os.path.splitext(filename)
    #     filename = prefix + ''
    
    with open(filename,  mode='a', newline='') as f:
        writer = csv.writer(f, delimiter =';')
        writer.writerow(arr)
