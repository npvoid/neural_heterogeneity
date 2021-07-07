import torch
import torch.nn as nn
import numpy as np


def loss(m, layer_recs, y_local, N_samp, prms):
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss(reduction='sum')

    log_p_y = log_softmax_fn(m)

    # Compute loss
    loss_low_spk = low_spk_count_loss(layer_recs, prms, N_samp)
    loss_up_spk = up_spk_count_loss(layer_recs, prms, N_samp)
    loss_val = loss_fn(log_p_y, y_local)/N_samp + loss_low_spk + loss_up_spk

    return loss_val


def low_spk_count_loss(layer_recs, prms, N_samp):
    sl = prms['sl']
    if not sl:
        return 0.

    thetal = prms['thetal']
    T = prms['nb_steps']
    # N_samp = prms['batch_size']
    N = sum([layer[0].shape[2] for layer in layer_recs])  # Total number of neurons in the network

    L1_batch = 0.
    for layer in layer_recs:
        if len(layer) == 3:
            tmp = (torch.clamp((1/T)*torch.sum(layer[-1], 1)-thetal, min=0.))**2
            L1_batch += torch.sum(tmp, (0, 1))
        else:
            continue
    return (sl/(N_samp+N))*L1_batch


def up_spk_count_loss(layer_recs, prms, N_samp):
    su = prms['su']
    if not su:
        return 0.

    thetau = prms['thetau']
    # N_samp = prms['batch_size']

    L2_batch = 0.
    for layer in layer_recs:
        if len(layer) == 3:
            N = layer[-1].shape[2]
            tmp = (torch.clamp((1/N)*torch.sum(layer[-1], (1, 2))-thetau, min=0.))**2
            L2_batch += torch.sum(tmp)
        else:
            continue
    return (su/N_samp)*L2_batch


def l2_weight_loss(model, prms):
    weight_decay = prms['weight_decay']
    l2_reg = 0
    for param in model.parameters():
        if not np.any(np.array(param.shape) == 1):
            l2_reg += 0.5*param.norm(2)**2
    l2_reg *= weight_decay
    return l2_reg
