import os
import numpy as np
import pickle
import torch
from contextlib import contextmanager


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def load_results(path_results, dirName, fileName):
    pickle_out = open(os.path.join(path_results, dirName, fileName), "rb")
    parameters = pickle.load(pickle_out)
    pickle_out.close()
    return parameters


def save_results(path_results, dirName, parameters):
    fileName = 'parameters.pickle'
    pickle_out = open(os.path.join(path_results, dirName, fileName), "wb")
    pickle.dump(parameters, pickle_out)
    pickle_out.close()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path_results, dirName, epoch, model, optimizer, train_loss_v, test_loss_v, train_acc_v, test_acc_v):

    last_model_filename = "model_last.pth"
    last_model_out_path = os.path.join(path_results, dirName, last_model_filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_v': train_loss_v,
        'test_loss_v': test_loss_v,
        'train_acc_v': train_acc_v,
        'test_acc_v': test_acc_v,
    }, last_model_out_path)
