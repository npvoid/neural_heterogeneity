import numpy as np
import pylab as plt
import seaborn as sns
import pickle as pckl
import os
from collections import defaultdict
import argparse

class Experiment:
    def __init__(self, nb_trials, nb_epochs, alpha=0.3, timescale=None):

        self.het_ab = ['hom', 'het']
        self.train_ab = ['no_ab', 'hom_ab', 'het_ab']
        self.lr = None
        self.nb_trials = nb_trials
        self.timescale = timescale

        # self.labels = defaultdict(dict)
        # self.labels['hom']['no_ab'] = 'Homogeneous'
        # self.labels['het']['no_ab'] = 'Heterogeneous'
        # self.labels['hom']['het_ab'] = 'Homogeneous + Heterogeneous Train'
        # self.labels['het']['het_ab'] = 'Heterogeneous + Heterogeneous Train'
        # self.labels['hom']['hom_ab'] = 'Homogeneous + Homogeneous Train'
        self.labels = defaultdict(dict)
        self.labels['hom']['no_ab'] = 'HomInit-StdTr'
        self.labels['het']['no_ab'] = 'HetInit-StdTr'
        self.labels['hom']['het_ab'] = 'HomInit-HetTr'
        self.labels['het']['het_ab'] = 'HetInit-HetTr'
        self.labels['hom']['hom_ab'] = 'HomInit-HomTr'

        self.epoch_idx = np.arange(nb_epochs) + 1

        self.train_loss = defaultdict(dict)
        self.test_loss = defaultdict(dict)
        self.train_acc = defaultdict(dict)
        self.test_acc = defaultdict(dict)

        self.train_loss_mean = defaultdict(dict)
        self.test_loss_mean = defaultdict(dict)
        self.train_acc_mean = defaultdict(dict)
        self.test_acc_mean = defaultdict(dict)

        self.train_loss_std = defaultdict(dict)
        self.test_loss_std = defaultdict(dict)
        self.train_acc_std = defaultdict(dict)
        self.test_acc_std = defaultdict(dict)

        self.train_loss_median = defaultdict(dict)
        self.test_loss_median = defaultdict(dict)
        self.train_acc_median = defaultdict(dict)
        self.test_acc_median = defaultdict(dict)

        self.train_loss_min = defaultdict(dict)
        self.test_loss_min = defaultdict(dict)
        self.train_acc_min = defaultdict(dict)
        self.test_acc_min = defaultdict(dict)

        self.train_loss_max = defaultdict(dict)
        self.test_loss_max = defaultdict(dict)
        self.train_acc_max = defaultdict(dict)
        self.test_acc_max = defaultdict(dict)

        for h in self.het_ab:
            for t in self.train_ab:
                self.train_loss[h][t] = np.zeros((nb_trials, nb_epochs))
                self.test_loss[h][t] = np.zeros((nb_trials, nb_epochs))
                self.train_acc[h][t] = np.zeros((nb_trials, nb_epochs))
                self.test_acc[h][t] = np.zeros((nb_trials, nb_epochs))

        self.alpha = alpha  # Transparency of fill area

        self.MEAN_train_acc = defaultdict(dict)
        self.MEAN_test_acc = defaultdict(dict)
        self.STD_train_acc = defaultdict(dict)
        self.STD_test_acc = defaultdict(dict)
        self.MEDIAN_train_acc = defaultdict(dict)
        self.MEDIAN_test_acc = defaultdict(dict)
        self.MIN_train_acc = defaultdict(dict)
        self.MIN_test_acc = defaultdict(dict)
        self.MAX_train_acc = defaultdict(dict)
        self.MAX_test_acc = defaultdict(dict)

    def run_results(self, path_results):
        self.read_data(path_results)
        self.mean()
        self.std()
        self.median()
        self.minmax()

    def plot_results(self, ss_epoch):
        self.plot_experiment()
        self.print_results(ss_epoch)

    def read_data(self, path_results):
        for dirpath, _, files in os.walk(path_results):
            for f in files:
                if f.endswith(".pickle"):
                    pickle_in = open(os.path.join(dirpath, 'parameters.pickle'), "rb")
                    parameters = pckl.load(pickle_in)
                    prms = parameters['prms']
                    # print(prms['batch_size'])
                    train_loss = prms['train_loss']
                    test_loss = prms['test_loss']
                    train_acc = prms['train_acc_v']
                    test_acc = prms['test_acc_v']

                    if prms['sparse_data_generator'] == "sparse_data_generator_scale":
                        train_loss = train_loss[self.timescale]
                        test_loss = test_loss[self.timescale]
                        train_acc = train_acc[self.timescale]
                        test_acc = test_acc[self.timescale]
                    else:
                        self.timescale = None

                    self.lr = prms['lr']
                    if 'Heterogeneous' in prms.keys():
                        prms['het_ab'] = prms['Heterogeneous']
                    i = prms['seed']
                    if i > self.nb_trials-1:
                        break
                    if prms['train_ab'] == 0 and prms['het_ab'] == 0:
                        if 'train_hom_ab' in prms.keys() and prms['train_hom_ab'] == 1:
                            self.train_loss['hom']['hom_ab'][i, :] = train_loss
                            self.test_loss['hom']['hom_ab'][i, :] = test_loss
                            self.train_acc['hom']['hom_ab'][i, :] = train_acc
                            self.test_acc['hom']['hom_ab'][i, :] = test_acc
                        else:
                            self.train_loss['hom']['no_ab'][i, :] = train_loss
                            self.test_loss['hom']['no_ab'][i, :] = test_loss
                            self.train_acc['hom']['no_ab'][i, :] = train_acc
                            self.test_acc['hom']['no_ab'][i, :] = test_acc
                    elif prms['train_ab'] == 0 and prms['het_ab'] == 1:
                        self.train_loss['het']['no_ab'][i, :] = train_loss
                        self.test_loss['het']['no_ab'][i, :] = test_loss
                        self.train_acc['het']['no_ab'][i, :] = train_acc
                        self.test_acc['het']['no_ab'][i, :] = test_acc
                    elif prms['train_ab'] == 1 and prms['het_ab'] == 0:
                        self.train_loss['hom']['het_ab'][i, :] = train_loss
                        self.test_loss['hom']['het_ab'][i, :] = test_loss
                        self.train_acc['hom']['het_ab'][i, :] = train_acc
                        self.test_acc['hom']['het_ab'][i, :] = test_acc
                    elif prms['train_ab'] == 1 and prms['het_ab'] == 1:
                        self.train_loss['het']['het_ab'][i, :] = train_loss
                        self.test_loss['het']['het_ab'][i, :] = test_loss
                        self.train_acc['het']['het_ab'][i, :] = train_acc
                        self.test_acc['het']['het_ab'][i, :] = test_acc

    def mean(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.train_loss_mean[h][t] = np.mean(self.train_loss[h][t], axis=0)
                self.test_loss_mean[h][t] = np.mean(self.test_loss[h][t], axis=0)
                self.train_acc_mean[h][t] = np.mean(self.train_acc[h][t], axis=0)
                self.test_acc_mean[h][t] = np.mean(self.test_acc[h][t], axis=0)

    def std(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.train_loss_std[h][t] = np.std(self.train_loss[h][t], axis=0) / np.sqrt(self.nb_trials)
                self.test_loss_std[h][t] = np.std(self.test_loss[h][t], axis=0)/ np.sqrt(self.nb_trials)
                self.train_acc_std[h][t] = np.std(self.train_acc[h][t], axis=0)/ np.sqrt(self.nb_trials)
                self.test_acc_std[h][t] = np.std(self.test_acc[h][t], axis=0)/ np.sqrt(self.nb_trials)

    def median(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.train_loss_median[h][t] = np.median(self.train_loss[h][t], axis=0)
                self.test_loss_median[h][t] = np.median(self.test_loss[h][t], axis=0)
                self.train_acc_median[h][t] = np.median(self.train_acc[h][t], axis=0)
                self.test_acc_median[h][t] = np.median(self.test_acc[h][t], axis=0)

    def minmax(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.train_loss_min[h][t] = np.min(self.train_loss[h][t], axis=0)
                self.test_loss_min[h][t] = np.min(self.test_loss[h][t], axis=0)
                self.train_acc_min[h][t] = np.min(self.train_acc[h][t], axis=0)
                self.test_acc_min[h][t] = np.min(self.test_acc[h][t], axis=0)

                self.train_loss_max[h][t] = np.max(self.train_loss[h][t], axis=0)
                self.test_loss_max[h][t] = np.max(self.test_loss[h][t], axis=0)
                self.train_acc_max[h][t] = np.max(self.train_acc[h][t], axis=0)
                self.test_acc_max[h][t] = np.max(self.test_acc[h][t], axis=0)

    def plot_experiment(self):
        sns.set()
        ## mean +- std
        self.plot_meanstd(self.train_loss_mean, self.train_loss_std, ylabel='Loss',
                          title='Train Loss History (mean, std)')
        self.plot_meanstd(self.test_loss_mean, self.test_loss_std, ylabel='Loss', title='Test Loss History (mean, std)')
        self.plot_meanstd(self.train_acc_mean, self.train_acc_std, ylabel='Accuracy',
                          title='Training Accuracy History (mean, std)')
        self.plot_meanstd(self.test_acc_mean, self.test_acc_std, ylabel='Accuracy',
                          title='Testing Accuracy History (mean, std)')
        ## median min_max
        self.plot_minmedianmax(self.train_loss_min, self.train_loss_median, self.train_loss_max, ylabel='Loss',
                               title='Train Loss History (min, median, max)')
        self.plot_minmedianmax(self.test_loss_min, self.test_loss_median, self.test_loss_max, ylabel='Loss',
                               title='Test Loss History (min, median, max)')
        self.plot_minmedianmax(self.train_acc_min, self.train_acc_median, self.train_acc_max, ylabel='Accuracy',
                               title='Training Accuracy History (min, median, max)')
        self.plot_minmedianmax(self.test_acc_min, self.test_acc_median, self.test_acc_max, ylabel='Accuracy',
                               title='Testing Accuracy History (min, median, max)')

    def plot_meanstd(self, mean, std, ylabel='', title=''):
        _, ax = plt.subplots()
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(mean[h][t]):
                    ax.fill_between(self.epoch_idx, mean[h][t] - std[h][t], mean[h][t] + std[h][t],
                                    alpha=self.alpha)
                    ax.plot(self.epoch_idx, mean[h][t], label=self.labels[h][t])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.title(title)

    def plot_minmedianmax(self, min, median, max, ylabel='', title=''):
        # Train Loss hist
        _, ax = plt.subplots()
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(max[h][t]):
                    ax.fill_between(self.epoch_idx, min[h][t], max[h][t],
                                    alpha=self.alpha)
                    ax.plot(self.epoch_idx, median[h][t], label=self.labels[h][t])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.title(title)

    def steady_state(self, ss_epoch):
        for t in self.train_ab:
            for h in self.het_ab:
                self.MEAN_train_acc[h][t] = np.mean(self.train_acc_mean[h][t][ss_epoch:])
                self.MEAN_test_acc[h][t] = np.mean(self.test_acc_mean[h][t][ss_epoch:])
                self.STD_train_acc[h][t] = np.mean(self.train_acc_std[h][t][ss_epoch:])  # TODO: Mind average std, not overall std
                self.STD_test_acc[h][t] = np.mean(self.test_acc_std[h][t][ss_epoch:])

                self.MEDIAN_train_acc[h][t] = np.mean(self.train_acc_median[h][t][ss_epoch:])
                self.MEDIAN_test_acc[h][t] = np.mean(self.test_acc_median[h][t][ss_epoch:])
                self.MIN_train_acc[h][t] = np.mean(self.train_acc_min[h][t][ss_epoch:])
                self.MIN_test_acc[h][t] = np.mean(self.test_acc_min[h][t][ss_epoch:])
                self.MAX_train_acc[h][t] = np.mean(self.train_acc_max[h][t][ss_epoch:])
                self.MAX_test_acc[h][t] = np.mean(self.test_acc_max[h][t][ss_epoch:])

    def print_results(self, ss_epoch):
        if self.timescale is not None:
            print('Timescale: {:g}'.format(self.timescale))
        self.steady_state(ss_epoch)
        print('Average Train Accuracy Steady State')
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(self.STD_train_acc[h][t]):
                    print("{:<25}: {:.3f} +- {:.3f}".format("Train "+self.labels[h][t], self.MEAN_train_acc[h][t], self.STD_train_acc[h][t]))
        print('Average Test Accuracy Steady State')
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(self.STD_test_acc[h][t]):
                    print("{:<25}: {:.3f} +- {:.3f}".format("Test "+self.labels[h][t], self.MEAN_test_acc[h][t], self.STD_test_acc[h][t]))
        print('Average Median Train Accuracy Steady State (min, median, max)')
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(self.MAX_train_acc[h][t]):
                    print("{:<25}: ({:.3f}, {:.3f}, {:.3f})".format("Train "+self.labels[h][t], self.MIN_train_acc[h][t],
                        self.MEDIAN_train_acc[h][t], self.MAX_train_acc[h][t]))
        print('Average Median Test Accuracy Steady State (min, median, max)')
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(self.MAX_test_acc[h][t]):
                    print("{:<25}: ({:.3f}, {:.3f}, {:.3f})".format("Test "+self.labels[h][t], self.MIN_test_acc[h][t],
                        self.MEDIAN_test_acc[h][t], self.MAX_test_acc[h][t]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network')
    parser.add_argument('--nb_seeds', type=int, default=10, help='Seed')
    parser.add_argument('--nb_epochs', type=int, default=75, help='Epochs')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha')
    parser.add_argument('--path_results', type=str, help='Results path')
    parser.add_argument('--timescale', type=float, default=1, help='Selecting time scale if you trained with multiple using sparse_data_generator_scale')

    prms = vars(parser.parse_args())

    ss_epoch = prms['nb_epochs']-1

    exp = Experiment(prms['nb_seeds'], prms['nb_epochs'], alpha=prms['alpha'], timescale=prms['timescale'])
    exp.run_results(prms['path_results'])
    exp.plot_results(ss_epoch)

    plt.show()


