from matplotlib import pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
import sys
sys.path.append("..")
from data_gen import open_file
from utils_plot import show2D

units, times, labels = open_file("dataset/train.h5")

data_idx = 1
timestamps = times[data_idx]
time_step = 1
time_step = 1e-6
print(labels)
print("Label:", labels[data_idx])

x_max = 34
y_max = 34
xaddr = units[data_idx] % x_max
yaddr = (units[data_idx] // x_max) % y_max
pol = units[data_idx] // (x_max*y_max)

pixel = (15, 15)
idx = (xaddr == pixel[0]) & (yaddr == pixel[1])
pol_plot = pol[idx]
ts_plot = (1e6 * timestamps[idx]).astype(int)

coo = [pol_plot, ts_plot]
i = torch.LongTensor(coo)
v = torch.FloatTensor(np.ones(len(coo[0])))

X_batch = torch.sparse.FloatTensor(i, v, torch.Size([2, int(6e6)])).to_dense()

show2D(timestamps, xaddr, yaddr, pol, time_step, frameRate=24, preComputeFrames=True, repeat=False, minTimeStamp=None, maxTimeStamp=30e-3)
