import numpy as np
import pylab as plt
import seaborn as sns
import pickle as pckl
import os
from collections import defaultdict
from results import Experiment


def plot_meanstd(x, mean, std, exp, xlabel='Learning Rate', ylabel='Accuracy', title=''):
    _, ax = plt.subplots()
    for t in exp.train_ab:
        for h in exp.het_ab:
            ax.fill_between(x, np.array(mean[h][t]) - np.array(std[h][t]), np.array(mean[h][t]) + np.array(std[h][t]), alpha=exp.alpha)
            ax.plot(x, mean[h][t], label=exp.labels[h][t])
    # plt.xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.title(title)


def plot_minmedianmax(x, min, median, max, exp, xlabel='Learning Rate', ylabel='Accuracy', title=''):
    # Train Loss hist
    _, ax = plt.subplots()
    for t in exp.train_ab:
        for h in exp.het_ab:
            ax.fill_between(x, min[h][t], max[h][t], alpha=exp.alpha)
            ax.plot(x, median[h][t], label=exp.labels[h][t])
    # plt.xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.title(title)


path_results = './SHD/results/robustness_neurs'
# dirs = ['0ab0het', '0ab1het', '1ab0het', '1ab1het']
dirs = ['0th0hth0ab0hab', '0th0hth0ab1hab', '0th0hth1ab0hab', '0th0hth1ab1hab']
nb_seeds = 1  # Number of trials
nb_epochs = 75
alpha = 0.3
ss_epoch = 60
lrs_dirs = os.listdir(path_results)

# Get data
exps = []
for lr_dir in lrs_dirs:
    exps.append(Experiment(nb_seeds, nb_epochs, alpha=alpha))
    exps[-1].run_results(os.path.join(path_results, lr_dir), dirs)
    exps[-1].steady_state(ss_epoch)

    # if lr_dir == '5.00e-03':
    #     print(lr_dir + ': ' + str(exps[-1].test_acc['het']['ab'][:, -15:]))
    #     print()
    #     print(lr_dir + ': ' + str(exps[-1].test_acc['het']['no_ab'][:, -15:]))
    #     print()
    #     print(lr_dir + ': ' + str(exps[-1].test_acc['hom']['ab'][:, -15:]))
    #     print()

lrs = np.array([float(lr) for lr in lrs_dirs])
ids = np.argsort(lrs)
lrs = lrs[ids]
exps = [exps[i] for i in ids]

# Process for plotting
LR_MEAN_train_acc = defaultdict(lambda: defaultdict(list))
LR_MEAN_test_acc = defaultdict(lambda: defaultdict(list))
LR_STD_train_acc = defaultdict(lambda: defaultdict(list))
LR_STD_test_acc = defaultdict(lambda: defaultdict(list))
LR_MEDIAN_train_acc = defaultdict(lambda: defaultdict(list))
LR_MEDIAN_test_acc = defaultdict(lambda: defaultdict(list))
LR_MIN_train_acc = defaultdict(lambda: defaultdict(list))
LR_MIN_test_acc = defaultdict(lambda: defaultdict(list))
LR_MAX_train_acc = defaultdict(lambda: defaultdict(list))
LR_MAX_test_acc = defaultdict(lambda: defaultdict(list))
het_ab = ['hom', 'het']
train_ab = ['no_ab', 'ab']
for i, lr_dir in enumerate(lrs_dirs):
    for t in train_ab:
        for h in het_ab:
            LR_MEAN_train_acc[h][t].append(exps[i].MEAN_train_acc[h][t])
            LR_MEAN_test_acc[h][t].append(exps[i].MEAN_test_acc[h][t])
            LR_STD_train_acc[h][t].append(exps[i].STD_train_acc[h][t])
            LR_STD_test_acc[h][t].append(exps[i].STD_test_acc[h][t])
            LR_MEDIAN_train_acc[h][t].append(exps[i].MEDIAN_train_acc[h][t])
            LR_MEDIAN_test_acc[h][t].append(exps[i].MEDIAN_test_acc[h][t])
            LR_MIN_train_acc[h][t].append(exps[i].MIN_train_acc[h][t])
            LR_MIN_test_acc[h][t].append(exps[i].MIN_test_acc[h][t])
            LR_MAX_train_acc[h][t].append(exps[i].MAX_train_acc[h][t])
            LR_MAX_test_acc[h][t].append(exps[i].MAX_test_acc[h][t])

# lrs = range(9)  # TODO: Change to better x-axis scaling
# Plot
sns.set()
## mean +- std
plot_meanstd(lrs, LR_MEAN_train_acc, LR_STD_train_acc, exps[-1], title='Training Accuracy Robustness (mean, std)')
plot_meanstd(lrs, LR_MEAN_test_acc, LR_STD_test_acc, exps[-1], title='Testing Accuracy Robustness (mean, std)')
## median min_max
plot_minmedianmax(lrs, LR_MIN_train_acc, LR_MEDIAN_train_acc, LR_MAX_train_acc, exps[-1], title='Training Accuracy Robustness (min, median, max)')
plot_minmedianmax(lrs, LR_MIN_test_acc, LR_MEDIAN_test_acc, LR_MAX_test_acc, exps[-1], title='Testing Accuracy Robustness (min, median, max)')
        
        




plt.show()




