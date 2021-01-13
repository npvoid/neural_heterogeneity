import torch
import importlib

from utils import *
from utils_plot import *
from reg_loss import loss
from clipper import *
from data_gen import open_file, get_mini_batch
from SuSpike import SuSpike
spike_fn = SuSpike.apply

from tqdm import tqdm
import argparse
import sys
import numpy as np
from model_summary import summary
from contextlib import redirect_stdout

# ------------------------------------------------ Parameters ---------------------------------------------------- #
parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network')
# Dataset
parser.add_argument('--seed', type=int, default=1000, help='Seed')
parser.add_argument('--dataset', type=str, default='SHD', help='Choose Dataset to train')
parser.add_argument('--train_path', type=str, default='dataset/train.h5', help='Training dataset file path')
parser.add_argument('--test_path', type=str, default='dataset/test.h5', help='Test dataset file path')
parser.add_argument('--results_path', type=str, default='results/', help='Save results file path')
parser.add_argument('--class_list', type=int, nargs='+', default=[])
# Simulation
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')  # 1e-3
parser.add_argument('--lr_ab', type=float, default=1e-3, help='Learning Rate of taus (alpha and betas)')
parser.add_argument('--betas', type=float, default=(0.9, 0.999), help='Betas for optimizer')  # (0.9, 0.999)
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay')  # 0.
parser.add_argument('--nb_epochs', type=int, default=150, help='Number of epochs')  # >100
parser.add_argument('--drop_last', type=bool, default=True, help='Drop the last (incomplete) batch in training')  # >100
# Regularization
parser.add_argument('--sl', type=float, default=0., help='l1 spike loss')  # 1.
parser.add_argument('--thetal', type=float, default=0., help='l1 spike loss')  # 0.01
parser.add_argument('--su', type=float, default=0., help='l2 spike loss')  # 0.06
parser.add_argument('--thetau', type=float, default=0., help='l2 spike loss')  # 100.
parser.add_argument('--rate', type=float, default=0., help='Rate input Poisson noise')  # 1.6
parser.add_argument('--p_del', type=float, default=0., help='prob spike deletion')  # 0.0008
# Network layout
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')  # ??
parser.add_argument('--nb_inputs', type=int, default=700, help='Number of input neurons')  # 700
parser.add_argument('--nb_hidden', type=int, nargs='+', default=[], help='List of hidden neurons per layer')
parser.add_argument('--nb_recurrent', type=int, nargs='+', default=128, help='recurrent layer size')  # 128
parser.add_argument('--nb_outputs', type=int, default=20, help='Number of output neurons')  # 20
parser.add_argument('--time_step', type=float, default=0.5e-3, help='Time step')  # 0.5e-3
parser.add_argument('--nb_steps', type=int, default=2000, help='Number of time steps')  # ?? 1400ms at most (needs to be corrected with time_step)
# Neuron
parser.add_argument('--tau_syn', type=float, default=10e-3, help='time constant for synapse')  # 10e-3
parser.add_argument('--tau_mem', type=float, default=20e-3, help='time constant for membrane potential')  # 20e-3
parser.add_argument('--threshold', type=float, default=1., help='spiking threshold')  # 1.
parser.add_argument('--tref', type=float, default=0., help='refractory time')  # 0
parser.add_argument('--gamma_shape', type=float, default=3, help='shape of Gamma distribution')
# Training Setup
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--train_ab', type=int, default=0, help='train heterogeneous time constants')
parser.add_argument('--het_ab', type=int, default=0, help='initialise with heterogeneous time constants')
parser.add_argument('--train_hom_ab', type=int, default=0, help='train homogeneous time constants')
parser.add_argument('--train_th', type=int, default=0, help='train threshold')
parser.add_argument('--het_th', type=int, default=0, help='initialise with heterogeneous thresholds')
parser.add_argument('--train_reset', type=int, default=0, help='train reset potentials')
parser.add_argument('--het_reset', type=int, default=0, help='initialise with heterogeneous reset potentials')
parser.add_argument('--train_rest', type=int, default=0, help='train rest potentials')
parser.add_argument('--het_rest', type=int, default=0, help='initialise with heterogeneous rest potentials')
parser.add_argument('--sparse_data_generator', type=str, default="sparse_data_generator", help='Choose the data generator')
parser.add_argument('--time_scale', type=float, nargs='+', default=[0.5, 1.0], help='Choose the time scale (Only apply to sparse_data_generator_scale)')
parser.add_argument('--model', type=str, default="RSNN", help='Choose the model to train')
parser.add_argument("--savestep", type=int, default=10, help="Sets saving step of model parameters")
parser.add_argument('--clip', type=int, default=1, help='clip alpha beta in range of 0 and 1')
parser.add_argument("--plot_step", type=int, default=50, help="Sets saving step of model parameters")

prms = vars(parser.parse_args())

# prms['w_scale'] = prms['w_scale'] * (1.0 - prms['beta'])
# prms['v_scale'] = prms['v_scale'] * (1.0 - prms['beta'])
# PyTorch
prms['dtype'] = torch.float
prms['device'] = torch.device("cuda") if prms['cuda'] else torch.device("cpu")
set_seed(prms['seed'])
print("Dataset:", prms['dataset'])
print("Device:", prms['device'])
print("Seed:", prms['seed'])
if prms['sparse_data_generator'] == "sparse_data_generator_scale":
    print("Time Scale: ", prms["time_scale"])

# Global Routines
sparse_data_generator = getattr(importlib.import_module("data_gen"), prms['sparse_data_generator'])
model_type = getattr(importlib.import_module("model"), prms['model'])
clipper = ZeroOneClipper()


def run(dir_save, dirName, prms, test_net=False):
    learn_prms, train_loss, test_loss, train_acc_v, test_acc_v = train_experiment(prms, dir_save, dirName, test_net=test_net)
    prms['train_loss'] = train_loss
    prms['test_loss'] = test_loss
    prms['train_acc_v'] = train_acc_v
    prms['test_acc_v'] = test_acc_v
    # Save results
    parameters = dict()
    parameters['prms'] = prms
    parameters['learn_prms'] = learn_prms
    save_results(dir_save, dirName, parameters)
    # Plot params and loss
    plot_param_dist(dir_save, dirName, prms, learn_prms)
    plot_loss(dir_save, dirName, prms)


def compute_classification_accuracy(units, times, labels, prms, model):
    """ Computes classification accuracy on supplied data in batches. """

    device = prms['device']

    num_samples = len(labels)  # Number of samples in data
    accs = 0
    with torch.no_grad():
        for batch in sparse_data_generator(units, times, labels, prms, shuffle=False, drop_last=False):
            x_local, y_local = batch[0].to(device), batch[1].to(device)
            layer_recs = model((0, 0, x_local))
            out = layer_recs[-1]
            m, _ = torch.max(out[1], 1)  # max over time
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = (y_local == am)  # compare to labels
            accs += tmp.sum().item()

    return accs/num_samples

def train_experiment(prms, dir_save, dirName, test_net=False):
    # Get Parameters
    lr = prms['lr']
    lr_ab = prms['lr_ab']
    betas = prms['betas']
    nb_epochs = prms['nb_epochs']
    batch_size = prms['batch_size']
    device = prms['device']
    prms['alpha'] = float(np.exp(-prms['time_step'] / prms['tau_syn']))
    prms['beta'] = float(np.exp(-prms['time_step'] / prms['tau_mem']))

    # Model and optimiser
    model = model_type(prms, rec=True).to(prms['device'])
    optimizer = torch.optim.Adam([
        {'params': [param for name, param in model.named_parameters() if 'alpha' not in name and 'beta' not in name]},
        {'params': [param for name, param in model.named_parameters() if 'alpha' in name or 'beta' in name],
         'lr': lr_ab}
    ], lr=lr, betas=betas)
    lambda_w = lambda epoch: 1
    lambda_ab = lambda epoch: 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_w, lambda_ab])

    with redirect_stdout(sys.stderr):
        summary(model, (prms['nb_steps'], prms['nb_inputs']))

    # ============================= For Parallel Multiple GPU Processing =====================================
    # if torch.cuda.device_count() > 1:
    #     with open('./results/results2.txt', 'a') as f:
    #         f.write("Let's use %d GPUs! \r\n" % torch.cuda.device_count())
    #     model = nn.DataParallel(model)

    # Load Checkpoint if exists
    file_path = os.path.join(dir_save, dirName, "model_last.pth")
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss_v = checkpoint['train_loss_v']
        test_loss_v = checkpoint['test_loss_v']
        train_acc_v = checkpoint['train_acc_v']
        test_acc_v = checkpoint['test_acc_v']
        print("=> loaded checkpoint '{} (iter {})'".format(file_path, start_epoch))
        if prms['sparse_data_generator'] == "sparse_data_generator_scale":
            print("Epoch {:d}".format(start_epoch))
            for i, ts in enumerate(prms['time_scale']):
                print("Timescale %g: Loss: train=%.4e, test=%.4e, Acc: train=%.3f, test=%.3f" % (
                    ts, train_loss_v[ts][-1], test_loss_v[ts][-1], train_acc_v[ts][-1], test_acc_v[ts][-1]))
        else:
            print("Epoch %i: Loss: train=%.4e, test=%.4e, Acc: train=%.4f, test=%.4f" % (
            start_epoch, train_loss_v[-1], test_loss_v[-1], train_acc_v[-1], test_acc_v[-1]))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))
        start_epoch = 0
        if prms['sparse_data_generator'] == "sparse_data_generator_scale":
            train_loss_v = dict()
            test_loss_v = dict()
            train_acc_v = dict()
            test_acc_v = dict()
            for ts in prms['time_scale']:
                train_loss_v[ts] = []
                test_loss_v[ts] = []
                train_acc_v[ts] = []
                test_acc_v[ts] = []
        else:
            # Loss
            train_loss_v = []
            test_loss_v = []
            train_acc_v = []
            test_acc_v = []

    if start_epoch == nb_epochs:
        learn_prms = [(p[0], p[1].detach().cpu().numpy()) for p in model.named_parameters()]
        return learn_prms, train_loss_v, test_loss_v, train_acc_v, test_acc_v

    # Obtain data samples and labels
    print(os.getcwd())
    print(os.path.join(os.getcwd(), prms['train_path']))
    units_train, times_train, labels_train = open_file(os.path.join(os.getcwd(), prms['train_path']))
    units_test, times_test, labels_test = open_file(os.path.join(os.getcwd(), prms['test_path']))

    if not prms['class_list']:
        prms['class_list'] = np.unique(labels_train).tolist()
    print("Class list:", prms['class_list'])

    num_train_samples = len(np.where(np.isin(labels_train, prms['class_list']))[0])
    num_test_samples = len(np.where(np.isin(labels_test, prms['class_list']))[0])

    if prms['drop_last']:
        num_train_samples = (num_train_samples // batch_size) * batch_size

    print("Number of training data:", num_train_samples)
    print("Number of test data:", num_test_samples)

    pbar = tqdm(total=nb_epochs, initial=start_epoch, desc=str(prms['train_ab'])+'ab_'+str(prms['het_ab'])+'het')

    # ===================== Can be used for faster implementation if enough memory is available ======================
    if prms['sparse_data_generator'] == "sparse_data_generator_fast":
        training_data_loader = sparse_data_generator(units_train, times_train, labels_train, prms, shuffle=True, epoch=start_epoch+1, drop_last=prms['drop_last'])
        testing_data_loader = sparse_data_generator(units_test, times_test, labels_test, prms, shuffle=False, epoch=start_epoch+1, drop_last=False)
    # ================================================================================================================

    # Training
    for epoch in range(start_epoch + 1, nb_epochs + 1):

        if prms['sparse_data_generator'] == "sparse_data_generator_scale":
            batch_train_loss = np.zeros(len(prms['time_scale']))
            batch_train_acc = np.zeros(len(prms['time_scale']))
            num_data_train_ts = np.zeros(len(prms['time_scale']))
        else:
            batch_train_loss = 0
            batch_train_acc = 0

        if prms['sparse_data_generator'] != "sparse_data_generator_fast":
            training_data_loader = sparse_data_generator(units_train, times_train, labels_train, prms, shuffle=True, epoch=epoch, drop_last=prms['drop_last'])

        if test_net:
            break

        for b, batch in enumerate(training_data_loader):
            # if b>1:
            #     break
            x_local, y_local = batch[0].to(device), batch[1].to(device)
            # Forward Pass
            layer_recs = model((0, 0, x_local))
            out = layer_recs[-1]
            m, _ = torch.max(out[1], 1)
            _, am = torch.max(m, 1)  # argmax over output units
            acc_val = (y_local == am)  # compare to labels

            # Compute Loss
            loss_val = loss(m, layer_recs, y_local, num_train_samples, prms)

            # Backward pass
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            if prms['clip']:
                model.apply(clipper)

            if prms['sparse_data_generator'] == "sparse_data_generator_scale":
                # Save for different timescales
                timescale_batch = batch[2]
                for i, ts in enumerate(prms['time_scale']):
                    idx_ts = (timescale_batch == ts)
                    if idx_ts.any():
                        num_data_train_ts[i] += idx_ts.sum()
                        batch_train_loss[i] += loss(m[idx_ts], None, y_local[idx_ts], 1, prms).item()
                        batch_train_acc[i] += acc_val[idx_ts].sum().item()
            else:
                batch_train_loss += loss_val.item()
                batch_train_acc += acc_val.sum().item()

            if b == 0 and ((epoch % prms['plot_step']) == 0 or epoch == 1):
                layer_recs_train_v = [(epoch, layer_recs)]
                plot_raster(dir_save, dirName, prms, layer_recs_train_v, mode='train', batch=0)

        if prms['sparse_data_generator'] == "sparse_data_generator_scale":
            for i, ts in enumerate(prms['time_scale']):
                mean_train_loss = batch_train_loss[i]/num_data_train_ts[i]
                mean_train_acc = batch_train_acc[i]/num_data_train_ts[i]
                train_loss_v[ts].append(mean_train_loss)
                train_acc_v[ts].append(mean_train_acc)
        else:
            # mean_train_loss = batch_train_loss/num_train_samples
            mean_train_loss = batch_train_loss
            mean_train_acc = batch_train_acc / num_train_samples
            train_loss_v.append(mean_train_loss)
            train_acc_v.append(mean_train_acc)

        # Testing
        if prms['sparse_data_generator'] == "sparse_data_generator_scale":
            batch_test_loss = np.zeros(len(prms['time_scale']))
            batch_test_acc = np.zeros(len(prms['time_scale']))
            num_data_test_ts = np.zeros(len(prms['time_scale']))
        else:
            batch_test_loss = 0
            batch_test_acc = 0

        if prms['sparse_data_generator'] != "sparse_data_generator_fast":
            testing_data_loader = sparse_data_generator(units_test, times_test, labels_test, prms, shuffle=False, epoch=epoch, drop_last=False)

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
                if prms['sparse_data_generator'] == "sparse_data_generator_scale":
                    # Save for different timescales
                    timescale_batch = batch[2]
                    for i, ts in enumerate(prms['time_scale']):
                        idx_ts = (timescale_batch == ts)
                        if idx_ts.any():
                            num_data_test_ts[i] += idx_ts.sum()
                            batch_test_loss[i] += loss(m[idx_ts], None, y_local[idx_ts], 1, prms).item()
                            batch_test_acc[i] += acc_val[idx_ts].sum().item()
                else:
                    loss_val = loss(m, layer_recs, y_local, num_test_samples, prms)
                    batch_test_loss += loss_val.item()
                    batch_test_acc += acc_val.sum().item()

                if b == 0 and ((epoch % prms['plot_step']) == 0 or epoch == 1):
                    layer_recs_test_v = [(epoch, layer_recs)]
                    plot_raster(dir_save, dirName, prms, layer_recs_test_v, mode='test', batch=0)

            if prms['sparse_data_generator'] == "sparse_data_generator_scale":
                for i, ts in enumerate(prms['time_scale']):
                    mean_test_loss = batch_test_loss[i] / num_data_test_ts[i]
                    mean_test_acc = batch_test_acc[i] / num_data_test_ts[i]
                    test_loss_v[ts].append(mean_test_loss)
                    test_acc_v[ts].append(mean_test_acc)
            else:
                # mean_test_loss = batch_test_loss / num_test_samples
                mean_test_loss = batch_test_loss
                mean_test_acc = batch_test_acc / num_test_samples
                test_loss_v.append(mean_test_loss)
                test_acc_v.append(mean_test_acc)

        if prms['sparse_data_generator'] == "sparse_data_generator_scale":
            pbar.write("Epoch {:d}".format(epoch))
            for i, ts in enumerate(prms['time_scale']):
                pbar.write("Timescale %g: Loss: train=%.4e, test=%.4e, Acc: train=%.3f, test=%.3f" % (
                ts, train_loss_v[ts][-1], test_loss_v[ts][-1], train_acc_v[ts][-1], test_acc_v[ts][-1]), file=sys.stderr)
        else:
            pbar.write("Epoch %i: Loss: train=%.4e, test=%.4e, Acc: train=%.3f, test=%.3f" % (epoch, train_loss_v[-1], test_loss_v[-1], train_acc_v[-1], test_acc_v[-1]), file=sys.stderr)

        #Save checkpoint
        if epoch % prms['savestep'] == 0 or epoch == nb_epochs:
            save_checkpoint(dir_save, dirName, epoch, model, optimizer, train_loss_v, test_loss_v, train_acc_v, test_acc_v)
            pbar.write("Checkpoint saved")

        # scheduler.step()
        # # for g in optimizer.param_groups:
        # #     print(g['lr'])
        pbar.update(1)
    pbar.close()

    if test_net:
        x_batch, y_batch = get_mini_batch(training_data_loader, device)
        layer_recs = model((0, 0, x_batch))
        plot_input(x_batch, y_batch)
        plot_layers(layer_recs)
        train_acc_v.append(0.)
        test_acc_v.append(0.)

    learn_prms = [(p[0], p[1].detach().cpu().numpy()) for p in model.named_parameters()]

    return learn_prms, train_loss_v, test_loss_v, train_acc_v, test_acc_v


def main():
    test_net = False
    with cd(prms['dataset']):
        dir_save = prms['results_path']
        dirStr = "{}{}th_{}{}{}ab_{}{}rest_{}{}reset".format(prms['train_th'], prms['het_th'],
                                                             prms['train_ab'], prms['het_ab'], prms['train_hom_ab'],
                                                             prms['train_rest'], prms['het_rest'],
                                                             prms['train_reset'], prms['het_reset'])
        dirName = os.path.join(dirStr, 'seed' + str(prms['seed']))

        if not os.path.exists(os.path.join(dir_save, dirName)):
            os.makedirs(os.path.join(dir_save, dirName))
            print("Directory", os.path.join(dir_save, dirName), "Created ")
        else:
            print("Directory", os.path.join(dir_save, dirName), "already exists")

        # Run
        run(dir_save, dirName, prms, test_net=test_net)



if __name__ == "__main__":
    main()
    # plt.show()