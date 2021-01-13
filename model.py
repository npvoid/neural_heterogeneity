import torch.nn as nn

from layers import *
from RecordingSequential import RecordingSequential


def as_list(x):
    if type(x) is list:
        return x
    else:
        return [x]


class SNN(nn.Module):
    def __init__(self, prms, rec=False):
        super(SNN, self).__init__()
        nb_iunits = prms['nb_inputs']
        nb_hidden = as_list(prms['nb_hidden'])
        nb_ounits = prms['nb_outputs']

        module_list = []

        module_list.append(SpikingLayer(nb_iunits, nb_hidden[0], prms))

        if len(nb_hidden) > 1:
            for i in range(len(nb_hidden)-1):
                module_list.append(SpikingLayer(nb_hidden[i], nb_hidden[i+1], prms))

        module_list.append(MembraneLayer(nb_hidden[-1], nb_ounits, prms))

        if rec:
            self.network = RecordingSequential(module_list)
        else:
            self.network = nn.Sequential(*module_list)

    def forward(self, x):
        return self.network(x)


class SNN_mem(nn.Module):
    def __init__(self, prms, rec=False):
        super(SNN_mem, self).__init__()
        nb_iunits = prms['nb_inputs']
        nb_hidden = as_list(prms['nb_hidden'])
        nb_recurrent = prms['nb_recurrent']
        nb_ounits = prms['nb_outputs']

        module_list = []

        module_list.append(SpikingLayer(nb_iunits, nb_hidden[0], prms))

        if len(nb_hidden) > 1:
            for i in range(len(nb_hidden)-1):
                module_list.append(SpikingLayer(nb_hidden[i], nb_hidden[i+1], prms))

        module_list.append(RecurrentSpikingLayer(nb_hidden[-1], nb_recurrent, prms))
        module_list.append(MembraneLayer(nb_recurrent, nb_ounits, prms))

        if rec:
            self.network = RecordingSequential(module_list)
        else:
            self.network = nn.Sequential(*module_list)

    def forward(self, x):
        return self.network(x)


class RSNN(nn.Module):
    def __init__(self, prms, rec=False):
        super(RSNN, self).__init__()
        nb_iunits = prms['nb_inputs']
        nb_recurrent = as_list(prms['nb_recurrent'])
        nb_ounits = prms['nb_outputs']

        module_list = []

        module_list.append(RecurrentSpikingLayer(nb_iunits, nb_recurrent[0], prms))

        if len(nb_recurrent) > 1:
            for i in range(len(nb_recurrent)-1):
                module_list.append(RecurrentSpikingLayer(nb_recurrent[i], nb_recurrent[i+1], prms))

        module_list.append(MembraneLayer(nb_recurrent[-1], nb_ounits, prms))

        if rec:
            self.network = RecordingSequential(module_list)
        else:
            self.network = nn.Sequential(*module_list)

    def forward(self, x):
        return self.network(x)

