import numpy as np
import pylab as plt
import seaborn as sns
import os
from collections import defaultdict
from data_gen import sparse_data_generator_scale
from data_gen import open_file
from utils import cd
import importlib
import torch
from reg_loss import loss
from datetime import datetime
from tqdm import tqdm
import argparse
import dill as pckl

from os.path import join as pjoin

class Experiment:
    def __init__(self, nb_trials, timescale, batch_size, alpha, device="cuda"):

        self.het_ab = ['hom', 'het']
        self.train_ab = ['no_ab', 'hom_ab', 'het_ab']
        self.lr = None
        self.nb_trials = nb_trials
        self.timescale = np.arange(timescale[0], timescale[1], timescale[2]).tolist()
        self.timescale_run = np.arange(timescale[0], timescale[1], timescale[2]).tolist()
        print(self.timescale)

        self.device = device
        self.batch_size = batch_size

        self.fig_suffix = "[" + ",".join(map(str, timescale)) + "].png"
        self.results_suffix = "[" + ",".join(map(str, timescale)) + "].pk"

        self.labels = defaultdict(dict)
        self.labels['hom']['no_ab'] = 'HomInit-StdTr'
        self.labels['het']['no_ab'] = 'HetInit-StdTr'
        self.labels['hom']['het_ab'] = 'HomInit-HetTr'
        self.labels['het']['het_ab'] = 'HetInit-HetTr'
        self.labels['hom']['hom_ab'] = 'HomInit-HomTr'

        self.test_loss = defaultdict(lambda : defaultdict(dict))
        self.test_acc = defaultdict(lambda : defaultdict(dict))

        self.test_loss_mean = defaultdict(dict)
        self.test_acc_mean = defaultdict(dict)

        self.test_loss_std = defaultdict(dict)
        self.test_acc_std = defaultdict(dict)

        self.test_loss_median = defaultdict(dict)
        self.test_acc_median = defaultdict(dict)

        self.test_loss_min = defaultdict(dict)
        self.test_acc_min = defaultdict(dict)

        self.test_loss_max = defaultdict(dict)
        self.test_acc_max = defaultdict(dict)

        for h in self.het_ab:
            for t in self.train_ab:
                for ts in self.timescale:
                    self.test_loss[h][t][ts] = np.zeros(nb_trials)
                    self.test_acc[h][t][ts] = np.zeros(nb_trials)

        self.alpha = alpha  # Transparency of fill area

    def run_results(self, path_results):
        self.load_results(path_results)
        self.read_data(path_results)
        self.mean()
        self.std()
        self.median()
        self.minmax()
        self.print_results()
        self.plot_experiment(path_results)

    def load_results(self, path_results):
        for dirpath, _, files in os.walk(path_results):
            for f in files:
                if f.endswith(".pk"):
                    filename = pjoin(path_results, f)
                    pickle_in = open(filename, "rb")
                    parameters = pckl.load(pickle_in)
                    file_ts = parameters['timescale']
                    for ts in file_ts:
                        if ts in self.timescale_run:
                            self.timescale_run.remove(ts)
                            for h in self.het_ab:
                                for t in self.train_ab:
                                    self.test_acc[h][t][ts] = parameters['test_acc'][h][t][ts]
                                    self.test_loss[h][t][ts] = parameters['test_loss'][h][t][ts]
        print(self.timescale_run)


    def read_data(self, path_results):
        DATA_LOADED = False

        pbar_tot = tqdm(total=40)

        for dirpath, _, files in os.walk(path_results):
            for f in files:
                if f.endswith(".pickle"):
                    pickle_in = open(os.path.join(dirpath, 'parameters.pickle'), "rb")
                    parameters = pckl.load(pickle_in)
                    prms = parameters['prms']
                    prms['train_th'] = False
                    prms['het_th'] = False
                    prms['train_reset'] = False
                    prms['het_reset'] = False
                    prms['train_rest'] = False
                    prms['het_rest'] = False
                    prms['train_hom_ab'] = False
                    nb_steps_org = prms['nb_steps']
                    # print(prms['train_ab'], prms['het_ab'])

                    device = self.device

                    if not DATA_LOADED:
                        with cd(prms['dataset']):
                            units_test, times_test, labels_test = open_file(
                                os.path.join(os.getcwd(), prms['test_path']))
                        DATA_LOADED = True
                    if not prms['class_list']:
                        prms['class_list'] = np.unique(labels_test).tolist()
                    num_test_samples = len(np.where(np.isin(labels_test, prms['class_list']))[0])
                    model_type = getattr(importlib.import_module("model"), prms['model'])

                    startTime = datetime.now()

                    for ts in self.timescale_run:
                        prms['time_scale'] = [ts]
                        prms['batch_size'] = int(self.batch_size//ts)
                        # prms['batch_size'] = self.batch_size

                        num_batches = -(-num_test_samples // prms['batch_size'])
                        prms['nb_steps'] = int(nb_steps_org*ts)
                        model = model_type(prms, rec=True).to(prms['device'])

                        testing_data_loader = sparse_data_generator_scale(units_test, times_test, labels_test, prms,
                                                                          shuffle=False, drop_last=False)

                        file_path = os.path.join(dirpath, "model_last.pth")
                        checkpoint = torch.load(file_path)

                        pretrained_dict = checkpoint['model_state_dict']
                        model_dict = model.state_dict()

                        # 1. filter out unnecessary keys
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        # print(pretrained_dict)
                        # 2. overwrite entries in the existing state dict
                        model_dict.update(pretrained_dict)
                        # print(model_dict)
                        # 3. load the new state dict
                        model.load_state_dict(model_dict)

                        batch_test_loss = 0
                        batch_test_acc = 0
                        # pbar = tqdm(total=num_batches)

                        with torch.no_grad():
                            for b, batch in enumerate(testing_data_loader):
                                x_local, y_local = batch[0].to(device), batch[1].to(device)
                                # Forward Pass
                                layer_recs = model((0, 0, x_local))
                                out = layer_recs[-1]
                                m, _ = torch.max(out[1], 1)
                                _, am = torch.max(m, 1)  # argmax over output units
                                acc_val = (y_local == am)  # compare to labels

                                # Compute Loss
                                loss_val = loss(m, layer_recs, y_local, num_test_samples, prms)
                                batch_test_loss += loss_val.item()
                                batch_test_acc += acc_val.sum().item()
                                # pbar.update(1)
                        # pbar.close()

                        test_loss = batch_test_loss
                        test_acc = batch_test_acc / num_test_samples

                        if 'Heterogeneous' in prms.keys():
                            prms['het_ab'] = prms['Heterogeneous']
                        i = prms['seed']
                        if i > self.nb_trials-1:
                            break
                        if prms['train_ab'] == 0 and prms['het_ab'] == 0:
                            if 'train_hom_ab' in prms.keys() and prms['train_hom_ab'] == 1:
                                self.test_loss['hom']['hom_ab'][ts][i] = test_loss
                                self.test_acc['hom']['hom_ab'][ts][i] = test_acc
                            else:
                                self.test_loss['hom']['no_ab'][ts][i] = test_loss
                                self.test_acc['hom']['no_ab'][ts][i] = test_acc
                        elif prms['train_ab'] == 0 and prms['het_ab'] == 1:
                            self.test_loss['het']['no_ab'][ts][i] = test_loss
                            self.test_acc['het']['no_ab'][ts][i] = test_acc
                        elif prms['train_ab'] == 1 and prms['het_ab'] == 0:
                            self.test_loss['hom']['het_ab'][ts][i] = test_loss
                            self.test_acc['hom']['het_ab'][ts][i] = test_acc
                        elif prms['train_ab'] == 1 and prms['het_ab'] == 1:
                            self.test_loss['het']['het_ab'][ts][i] = test_loss
                            self.test_acc['het']['het_ab'][ts][i] = test_acc
                    # self.save_results(path_results)
                    pbar_tot.update(1)
                    # print("It took {}".format(datetime.now() - startTime))
        pbar_tot.close()

    def mean(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.test_loss_mean[h][t] = np.zeros(len(self.timescale))
                self.test_acc_mean[h][t] = np.zeros(len(self.timescale))
                for j, ts in enumerate(self.timescale):
                    self.test_loss_mean[h][t][j] = np.mean(self.test_loss[h][t][ts])
                    self.test_acc_mean[h][t][j] = np.mean(self.test_acc[h][t][ts])

    def std(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.test_loss_std[h][t] = np.zeros(len(self.timescale))
                self.test_acc_std[h][t] = np.zeros(len(self.timescale))
                for j, ts in enumerate(self.timescale):
                    self.test_loss_std[h][t][j] = np.std(self.test_loss[h][t][ts])/ np.sqrt(self.nb_trials)
                    self.test_acc_std[h][t][j] = np.std(self.test_acc[h][t][ts])/ np.sqrt(self.nb_trials)

    def median(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.test_loss_median[h][t] = np.zeros(len(self.timescale))
                self.test_acc_median[h][t] = np.zeros(len(self.timescale))
                for j, ts in enumerate(self.timescale):
                    self.test_loss_median[h][t][j] = np.median(self.test_loss[h][t][ts])
                    self.test_acc_median[h][t][j] = np.median(self.test_acc[h][t][ts])

    def minmax(self):
        for h in self.het_ab:
            for t in self.train_ab:
                self.test_loss_min[h][t] = np.zeros(len(self.timescale))
                self.test_acc_min[h][t] = np.zeros(len(self.timescale))
                self.test_loss_max[h][t] = np.zeros(len(self.timescale))
                self.test_acc_max[h][t] = np.zeros(len(self.timescale))
                for j, ts in enumerate(self.timescale):
                    self.test_loss_min[h][t][j] = np.min(self.test_loss[h][t][ts])
                    self.test_acc_min[h][t][j] = np.min(self.test_acc[h][t][ts])

                    self.test_loss_max[h][t][j] = np.max(self.test_loss[h][t][ts])
                    self.test_acc_max[h][t][j] = np.max(self.test_acc[h][t][ts])

    def plot_experiment(self, path_results):
        sns.set()
        ## mean +- std
        self.plot_meanstd(self.test_loss_mean, self.test_loss_std, pjoin(path_results, "testloss_mean"+self.fig_suffix), ylabel='Loss', title='Test Loss at Different Timescales (mean, std)')
        self.plot_meanstd(self.test_acc_mean, self.test_acc_std, pjoin(path_results, "testacc_mean"+self.fig_suffix), ylabel='Accuracy',
                          title='Testing Accuracy at Different Timescales (mean, std)')
        ## median min_max
        self.plot_minmedianmax(self.test_loss_min, self.test_loss_median, self.test_loss_max, pjoin(path_results, "testloss_med"+self.fig_suffix), ylabel='Loss',
                               title='Test Loss at Different Timescales (min, median, max)')
        self.plot_minmedianmax(self.test_acc_min, self.test_acc_median, self.test_acc_max, pjoin(path_results, "testacc_med"+self.fig_suffix), ylabel='Accuracy',
                               title='Testing Accuracy at Different Timescales (min, median, max)')

    def plot_meanstd(self, mean, std, filepath, ylabel='', title=''):
        _, ax = plt.subplots()
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(mean[h][t]):
                    ax.fill_between(self.timescale, mean[h][t] - std[h][t], mean[h][t] + std[h][t],
                                    alpha=self.alpha)
                    ax.plot(self.timescale, mean[h][t], label=self.labels[h][t])
        ax.set_xlabel('Timescale')
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.title(title)
        plt.savefig(filepath)

    def plot_minmedianmax(self, min, median, max, filepath, ylabel='', title=''):
        # Train Loss hist
        _, ax = plt.subplots()
        for t in self.train_ab:
            for h in self.het_ab:
                if np.count_nonzero(max[h][t]):
                    ax.fill_between(self.timescale, min[h][t], max[h][t],
                                    alpha=self.alpha)
                    ax.plot(self.timescale, median[h][t], label=self.labels[h][t])
        ax.set_xlabel('Timescale')
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.title(title)
        plt.savefig(filepath)

    def save_results(self, path_results):
        parameters = dict()
        parameters['test_acc'] = self.test_acc
        parameters['test_loss'] = self.test_loss
        parameters['timescale'] = self.timescale

        if not os.path.exists(pjoin(path_results, self.results_suffix)):
            pickle_out = open(pjoin(path_results, self.results_suffix), "wb")
            pckl.dump(parameters, pickle_out)
            pickle_out.close()

    def print_results(self):
        for i, ts in enumerate(self.timescale):
            print('Timescale: {:g}'.format(ts))
            print('Average Test Accuracy')
            for t in self.train_ab:
                for h in self.het_ab:
                    if np.count_nonzero(self.test_acc_std[h][t][i]):
                        print("{:<25}: {:.3f} +- {:.3f}".format("Test "+self.labels[h][t], self.test_acc_mean[h][t][i], self.test_acc_std[h][t][i]))
            print('Average Median Test Accuracy (min, median, max)')
            for t in self.train_ab:
                for h in self.het_ab:
                    if np.count_nonzero(self.test_acc_max[h][t][i]):
                        print("{:<25}: ({:.3f}, {:.3f}, {:.3f})".format("Test "+self.labels[h][t], self.test_acc_min[h][t][i], self.test_acc_median[h][t][i], self.test_acc_max[h][t][i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network')
    parser.add_argument('--nb_seeds', type=int, default=10, help='Seed')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--ts_start', type=float, help='Timescale start')
    parser.add_argument('--ts_end', type=float, help='Timescale end')
    parser.add_argument('--ts_step', type=float, help='Timescale step')
    parser.add_argument('--path_results', type=str, help='Results path')

    prms = vars(parser.parse_args())

    timescale = [prms['ts_start'], prms['ts_end'], prms['ts_step']]
    print(timescale)

    startTime = datetime.now()
    exp = Experiment(prms['nb_seeds'], timescale, prms['batch_size'], prms['alpha'])
    exp.run_results(prms['path_results'])
    exp.save_results(prms['path_results'])
    plt.show()
    print("It took {}".format(datetime.now() - startTime))



