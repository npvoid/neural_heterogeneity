from matplotlib import pyplot as plt
import seaborn as sns
import os
from utils_plot import ab2tau
import pickle as pckl
from collections import defaultdict
import numpy as np
import importlib

class Experiment_dist:
    def __init__(self, nb_trials, nb_layers=2):

        self.het_ab = ['hom', 'het']
        self.train_ab = ['no_ab', 'hom_ab', 'het_ab']
        self.pname =['alpha', 'beta']

        self.labels_dist = dict()
        self.labels_dist['alpha'] = r'$\tau_s$'
        self.labels_dist['beta'] = r'$\tau_m$'

        self.labels = defaultdict(dict)
        self.labels['hom']['no_ab'] = 'HomInit-StdTr'
        self.labels['het']['no_ab'] = 'HetInit-StdTr'
        self.labels['hom']['het_ab'] = 'HomInit-HetTr'
        self.labels['het']['het_ab'] = 'HetInit-HetTr'
        self.labels['hom']['hom_ab'] = 'HomInit-HomTr'

        self.nb_trials = nb_trials
        self.nb_layers = nb_layers

        self.time_step = None

        self.dist = defaultdict(lambda: defaultdict(dict))

        for h in self.het_ab:
            for t in self.train_ab:
                for p in self.pname:
                    self.dist[h][t][p] = []
                    for l in range(self.nb_layers):
                        self.dist[h][t][p].append([])



    def read_data(self, path_results, dirs):
        for d in dirs:
            for dirpath, _, files in os.walk(os.path.join(path_results, d)):
                for f in files:
                    if f.endswith(".pickle"):
                        pickle_in = open(os.path.join(dirpath, 'parameters.pickle'), "rb")
                        parameters = pckl.load(pickle_in)
                        prms = parameters['prms']
                        learn_prms = parameters['learn_prms']
                        self.time_step =prms['time_step']

                        if 'Heterogeneous' in prms.keys():
                            prms['het_ab'] = prms['Heterogeneous']
                        i = prms['seed']
                        if i > self.nb_trials - 1:
                            break

                        for lparam_name, lparam in learn_prms:
                            layer = int(lparam_name.replace('.', ' ').split()[1])  # Layer number
                            pname = lparam_name.replace('.', ' ').split()[2]  # Learned Parameter name

                            if pname in self.pname and layer in range(self.nb_layers):
                                if prms['train_ab'] == 0 and prms['het_ab'] == 0:
                                    if 'train_hom_ab' in prms.keys() and prms['train_hom_ab'] == 1:
                                        self.dist['hom']['hom_ab'][pname][layer].extend(lparam.flatten())
                                    else:
                                        self.dist['hom']['no_ab'][pname][layer].extend(lparam.flatten())
                                elif prms['train_ab'] == 0 and prms['het_ab'] == 1:
                                    self.dist['het']['no_ab'][pname][layer].extend(lparam.flatten())
                                elif prms['train_ab'] == 1 and prms['het_ab'] == 0:
                                    self.dist['hom']['het_ab'][pname][layer].extend(lparam.flatten())
                                elif prms['train_ab'] == 1 and prms['het_ab'] == 1:
                                    self.dist['het']['het_ab'][pname][layer].extend(lparam.flatten())


    def plot_param_dist(self):
        sns.set_style('darkgrid')
        # Plot distribution of synaptic and membrane time constants

        for h in self.het_ab:
            for t in self.train_ab:
                for p in self.pname:
                    for layer in range(self.nb_layers):
                        if self.dist[h][t][p][layer]:
                            plt.figure()
                            # Time constants distribution
                            tau = ab2tau(self.dist[h][t][p][layer], self.time_step) / 1e-3 #in ms
                            ax = sns.distplot(tau, kde=False)
                            plt.xlabel('ms')
                            title = 'Layer {:d}: {} distribution ({})'.format(layer, self.labels_dist[p], self.labels[h][t])

                            ax.set_title(title)

    def run_dist(self, path_results, dirs):
        self.read_data(path_results, dirs)
        self.plot_param_dist()


if __name__ == "__main__":
    path_results = './NMNIST/results/nmnist_100ms'
    # path_results = './SHD/results/shd_train_hom'
    # path_results = './DVS/results/0reg0noise'
    dirs = ['00th_000ab_00rest_00reset0', '00th_010ab_00rest_00reset', '00th_100ab_00rest_00reset', '00th_110ab_00rest_00reset']
    # dirs = ['0ab0het', '0ab1het', '1ab0het', '1ab1het']
    # dirs = ['0th0hth0ab0hab', '0th1hth0ab0hab', '1th0hth0ab0hab', '1th1hth0ab0hab']
    # dirs = ['0th0hth0ab0hab', '0th0hth0ab1hab', '0th0hth1ab0hab', '0th0hth1ab1hab']
    # dirs = ['0ab0hab0thom', '0ab0hab1thom']
    nb_seeds = 10  # Number of trials

    exp = Experiment_dist(nb_seeds, 1)
    exp.run_dist(path_results, dirs)
    # exp.plot_results(ss_epoch)

    plt.show()