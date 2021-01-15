from matplotlib import pyplot as plt
import seaborn as sns
import os
from utils_plot import ab2tau
import pickle as pckl
from collections import defaultdict
import numpy as np
import importlib
import argparse

class Experiment_dist:
    def __init__(self, nb_trials, nb_layers):

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



    def read_data(self, path_results):
        for dirpath, _, files in os.walk(path_results):
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

    def run_dist(self, path_results):
        self.read_data(path_results)
        self.plot_param_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network')
    parser.add_argument('--nb_seeds', type=int, default=10, help='Seed')
    parser.add_argument('--nb_layers', type=int, default=1, help='Number of layers to be plotted')
    parser.add_argument('--path_results', type=str, help='Results path')

    prms = vars(parser.parse_args())

    exp = Experiment_dist(prms['nb_seeds'], prms['nb_layers'])
    exp.run_dist(prms['path_results'])

    plt.show()