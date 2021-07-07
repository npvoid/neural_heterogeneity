from matplotlib import pyplot as plt
import seaborn as sns
import torch
import numpy as np
import sys
sys.path.append("..")
from data_gen import open_file
from utils_plot import show2D

units, times, labels = open_file("dataset/train.h5")

data_idx = 12
print(labels[:])
print(labels[data_idx])
timestamps = times[data_idx]
# time_step = 1
time_step = 1e-6

xaddr = units[data_idx] & 0x0000007F
yaddr = (units[data_idx] >> 7) & 0x0000007F
pol = (units[data_idx] >> 14) & 0x00000001

show2D(timestamps, xaddr, yaddr, pol, time_step, frameRate=24, preComputeFrames=True, repeat=False, minTimeStamp=None, maxTimeStamp=5e-3)