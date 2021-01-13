from sklearn.metrics import confusion_matrix
import mlxtend.plotting as mlx
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import seaborn as sns
import pickle as pckl
import os
from collections import defaultdict
from data_gen import sparse_data_generator
import importlib
import torch
from datetime import datetime
from tqdm import tqdm

from utils import load_results, cd
from data_gen import open_file
from os.path import join as pjoin
import argparse
import matplotlib.ticker as plticker


def plot_confusion_matrix(multiclass, class_list=None):
    fig, ax = mlx.plot_confusion_matrix(conf_mat=multiclass,
                                                     colorbar=True,
                                                     show_absolute=False,
                                                     show_normed=True)
    loc = plticker.MultipleLocator(base=1.0)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    if class_list:
        ax.set_xticklabels([''] + class_list)
        ax.set_yticklabels([''] + class_list)

    # plt.show()

class Experiment:
    def __init__(self, nb_trials, batch_size, alpha, device="cuda"):

        self.het_ab = ['hom', 'het']
        self.train_ab = ['no_ab', 'het_ab']
        self.lr = None
        self.nb_trials = nb_trials

        self.device = device
        self.batch_size = batch_size

        self.fig_suffix = ".png"

        self.labels = defaultdict(dict)
        self.labels['hom']['no_ab'] = 'HomInit-StdTr'
        self.labels['het']['no_ab'] = 'HetInit-StdTr'
        self.labels['hom']['het_ab'] = 'HomInit-HetTr'
        self.labels['het']['het_ab'] = 'HetInit-HetTr'
        # self.labels['hom']['hom_ab'] = 'HomInit-HomTr'

        self.test_true = defaultdict(dict)
        self.test_pred = defaultdict(dict)

        for h in self.het_ab:
            for t in self.train_ab:
                self.test_true[h][t] = [[] for i in range(self.nb_trials)]
                self.test_pred[h][t] = [[] for i in range(self.nb_trials)]

        self.alpha = alpha  # Transparency of fill area

    def run_results(self, path_results):
        self.read_data(path_results)
        self.plot_experiment(path_results)

    def read_data(self, path_results):
        DATA_LOADED = False
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
                    # print(prms['train_ab'], prms['het_ab'])

                    device = self.device

                    if not DATA_LOADED:
                        with cd(prms['dataset']):
                            if os.path.exists(prms['test_path']):
                                units_test, times_test, labels_test = open_file(prms['test_path'])
                            else:
                                units_test, times_test, labels_test = open_file('dataset/test.h5')
                                print("Default test file path used")
                        DATA_LOADED = True

                    if self.batch_size is not None:
                        prms['batch_size'] = self.batch_size
                    if not prms['class_list']:
                        prms['class_list'] = np.unique(labels_test).tolist()
                    self.class_list = prms['class_list']
                    # print(self.class_list)
                    num_test_samples = len(np.where(np.isin(labels_test, prms['class_list']))[0])
                    num_batches = -(-num_test_samples // prms['batch_size'])
                    model_type = getattr(importlib.import_module("model"), prms['model'])
                    model = model_type(prms, rec=True).to(prms['device'])

                    testing_data_loader = sparse_data_generator(units_test, times_test, labels_test, prms,
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

                    # print(num_batches)
                    print(file_path)
                    if 'Heterogeneous' in prms.keys():
                        prms['het_ab'] = prms['Heterogeneous']
                    i = prms['seed']
                    if i > self.nb_trials - 1:
                        break

                    pbar = tqdm(total=num_batches)

                    with torch.no_grad():
                        for b, batch in enumerate(testing_data_loader):
                            x_local, y_local = batch[0].to(device), batch[1].to(device)
                            # Forward Pass
                            layer_recs = model((0, 0, x_local))
                            out = layer_recs[-1]
                            m, _ = torch.max(out[1], 1)
                            _, am = torch.max(m, 1)  # argmax over output units
                            if prms['train_ab'] == 0 and prms['het_ab'] == 0:
                                if 'train_hom_ab' in prms.keys() and prms['train_hom_ab'] == 1:
                                    self.test_true['hom']['hom_ab'][i].extend(y_local.tolist())
                                    self.test_pred['hom']['hom_ab'][i].extend(am.tolist())
                                else:
                                    self.test_true['hom']['no_ab'][i].extend(y_local.tolist())
                                    self.test_pred['hom']['no_ab'][i].extend(am.tolist())
                            elif prms['train_ab'] == 0 and prms['het_ab'] == 1:
                                self.test_true['het']['no_ab'][i].extend(y_local.tolist())
                                self.test_pred['het']['no_ab'][i].extend(am.tolist())
                            elif prms['train_ab'] == 1 and prms['het_ab'] == 0:
                                self.test_true['hom']['het_ab'][i].extend(y_local.tolist())
                                self.test_pred['hom']['het_ab'][i].extend(am.tolist())
                            elif prms['train_ab'] == 1 and prms['het_ab'] == 1:
                                self.test_true['het']['het_ab'][i].extend(y_local.tolist())
                                self.test_pred['het']['het_ab'][i].extend(am.tolist())
                            pbar.update(1)
                    pbar.close()

    def plot_experiment(self, path_results):
        # Train Loss hist
        for t in self.train_ab:
            for h in self.het_ab:
                true_labels = list(np.array(self.test_true[h][t]).flat)
                pred_labels = list(np.array(self.test_pred[h][t]).flat)
                mat = confusion_matrix(true_labels, pred_labels)
                # print(mat)
                plot_confusion_matrix(mat, class_list=self.class_list)
                plt.savefig(pjoin(path_results,self.labels[h][t]+self.fig_suffix))

    def save_results(self, path_results):
        parameters = dict()
        parameters['true'] = self.test_true
        parameters['pred'] = self.test_pred

        pickle_out = open(os.path.join(path_results, "true_pred_labels"), "wb")
        pckl.dump(parameters, pickle_out)
        pickle_out.close()

    def load_results(self, path_results):
        pickle_out = open(os.path.join(path_results, "true_pred_labels"), "rb")

        parameters = pckl.load(pickle_out)

        self.test_true = parameters['true']
        self.test_pred = parameters['pred']

        pickle_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network')
    parser.add_argument('--nb_seeds', type=int, default=10, help='Seed')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch Size')
    parser.add_argument('--path_results', type=str, help='Results path')

    prms = vars(parser.parse_args())

    startTime = datetime.now()
    exp = Experiment(prms['nb_seeds'], prms['batch_size'], prms['alpha'])
    exp.run_results(prms['path_results'])
    exp.save_results(prms['path_results'])
    print("It took {}".format(datetime.now() - startTime))
    plt.show()
