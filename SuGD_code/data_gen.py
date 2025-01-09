import os
import numpy as np
import torch
import tables
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import coo_matrix
from utils import set_seed


def open_file(hdf5_file_path):
    fileh = tables.open_file(hdf5_file_path, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels
    return units, times, labels


def poisson_spikes_gen(nb_units, nb_steps, rate, dt, seed):
    set_seed(seed)
    spike_trains = (np.random.uniform(0, 1, (nb_units, nb_steps)) <= rate * dt).astype(int)
    spike_trains = coo_matrix(spike_trains)
    times = spike_trains.col
    units = spike_trains.row
    return times, units


def poisson_spikes_indices(size, p, seed):
    set_seed(seed)
    idx = np.unique((np.arange(1, size+1)) * (np.random.uniform(size=size)>=p).astype(int)) - 1
    return idx


def sparse_data_generator(units, times, labels, prms, shuffle=True, epoch=0, drop_last=True):
    seed = prms['seed']+epoch
    set_seed(seed)
    batch_size = prms['batch_size']
    nb_steps = prms['nb_steps']
    nb_units = prms['nb_inputs']
    time_step = prms['time_step']
    rate = prms['rate']
    p_del = prms['p_del']
    class_list = prms['class_list']

    sample_index = np.where(np.isin(labels, class_list))[0]  # Indices to access the data_paths
    num_samples = len(sample_index)  # Number of samples in data

    if drop_last:
        number_of_batches = num_samples // batch_size  # Number of total batches with given data size
    else:
        number_of_batches = -(-num_samples // batch_size)

    if shuffle:
        np.random.shuffle(sample_index)  # Shuffle the paths indices

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:min(num_samples, batch_size * (counter + 1))]
        batch_size = len(batch_index)
        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            ts = (np.round(times[idx]*1./time_step).astype(np.int64))
            us = units[idx]

            # Constrain spike length
            idx = (ts < nb_steps)
            ts = ts[idx]
            us = us[idx]
            # Add random spikes
            ts_r, us_r = poisson_spikes_gen(nb_units, nb_steps, rate, time_step, seed)
            ts = np.concatenate((ts, ts_r))
            us = np.concatenate((us, us_r))
            # Delete spikes at random with given prob.
            idx = poisson_spikes_indices(ts.size, p_del, seed)
            ts = ts[idx]
            us = us[idx]

            batch = [bc for _ in range(ts.size)]
            coo[0].extend(batch)
            coo[1].extend(ts.tolist())
            coo[2].extend(us.tolist())

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse_coo_tensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to_dense()
        y_batch = torch.tensor([class_list.index(a) for a in labels[batch_index].astype(np.int64)], dtype=torch.long)

        X_batch[X_batch[:] > 1.] = 1.

        yield X_batch, y_batch

        counter += 1

def shd_augmented_sparse_data_generator(units, times, labels, prms, shuffle=True, epoch=0, drop_last=True):
    seed = prms['seed']+epoch
    set_seed(seed)
    batch_size = prms['batch_size']
    nb_steps = prms['nb_steps']
    nb_units = prms['nb_inputs']
    time_step = prms['time_step']
    rate = prms['rate']
    p_del = prms['p_del']
    class_list = prms['class_list']
    assert prms['nb_inputs']==70, "Number of input channels must be 70"

    sample_index = np.where(np.isin(labels, class_list))[0]  # Indices to access the data_paths
    num_samples = len(sample_index)  # Number of samples in data

    if drop_last:
        number_of_batches = num_samples // batch_size  # Number of total batches with given data size
    else:
        number_of_batches = -(-num_samples // batch_size)

    if shuffle:
        np.random.shuffle(sample_index)  # Shuffle the paths indices

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:min(num_samples, batch_size * (counter + 1))]
        batch_size = len(batch_index)
        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            ts = (np.round(times[idx]*1./time_step).astype(np.int))
            us = units[idx]

            # Add jitter
            jitter_noise = 20*np.random.randn(*us.shape)
            jitter_noise = np.rint(jitter_noise).astype(np.int)
            us = (us + jitter_noise) % 700
            # Merge channels
            us = us // 10

            # Constrain spike length
            idx = (ts < nb_steps)
            ts = ts[idx]
            us = us[idx]
            # Add random spikes
            ts_r, us_r = poisson_spikes_gen(nb_units, nb_steps, rate, time_step, seed)
            ts = np.concatenate((ts, ts_r))
            us = np.concatenate((us, us_r))
            # Delete spikes at random with given prob.
            idx = poisson_spikes_indices(ts.size, p_del, seed)
            ts = ts[idx]
            us = us[idx]

            batch = [bc for _ in range(ts.size)]
            coo[0].extend(batch)
            coo[1].extend(ts.tolist())
            coo[2].extend(us.tolist())

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to_dense()
        y_batch = torch.tensor([class_list.index(a) for a in labels[batch_index].astype(np.int)], dtype=torch.long)

        X_batch[X_batch[:] > 1.] = 1.

        yield X_batch, y_batch

        counter += 1


def get_mini_batch(training_data_loader, device):
    for batch in training_data_loader:
        x_local, y_local = batch[0].to(device), batch[1].to(device)
        return x_local, y_local

# ===================== Can be used for faster implementation if enough memory is available ======================

def sparse_data_generator_fast(units, times, labels, prms, shuffle=True, epoch=0, drop_last=True):
    seed = prms['seed']+epoch
    set_seed(seed)
    batch_size = prms['batch_size']
    nb_steps = prms['nb_steps']
    nb_units = prms['nb_inputs']
    time_step = prms['time_step']
    class_list = prms['class_list']

    sample_index = np.where(np.isin(labels, class_list))[0]  # Indices to access the data_paths
    num_samples = len(sample_index)  # Number of samples in data

    coo = [[] for i in range(3)]
    for bc, idx in enumerate(sample_index):
        ts = (np.round(times[idx]/time_step).astype(np.int))
        us = units[idx]

        # Constrain spike length
        idx = (ts < nb_steps)
        ts = ts[idx]
        us = us[idx]

        data_idx = [bc for _ in range(ts.size)]

        coo[0].extend(data_idx)
        coo[1].extend(ts.tolist())
        coo[2].extend(us.tolist())

    i = torch.LongTensor(coo)
    v = torch.FloatTensor(np.ones(len(coo[0])))

    X_batch = torch.sparse.FloatTensor(i, v, torch.Size([num_samples, nb_steps, nb_units])).to_dense()
    y_batch = torch.tensor([class_list.index(a) for a in labels[sample_index].astype(np.int)], dtype=torch.long)

    X_batch[X_batch[:] > 1.] = 1.

    dataset = TensorDataset(X_batch, y_batch)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return data_loader


def sparse_data_generator_scale(units, times, labels, prms, shuffle=True, epoch=0, drop_last=True):
    seed = prms['seed']+epoch
    set_seed(seed)
    batch_size = prms['batch_size']
    nb_steps = prms['nb_steps']
    nb_units = prms['nb_inputs']
    time_step = prms['time_step']
    rate = prms['rate']
    p_del = prms['p_del']
    class_list = prms['class_list']
    time_scale = np.array(prms['time_scale'])

    sample_index = np.where(np.isin(labels, class_list))[0]  # Indices to access the data_paths

    # extend for multiple timescale
    num_samples = len(sample_index)
    # num_samples = len(sample_index)*len(time_scale)  # Number of samples in data

    # Draw a random timescale factor from the list
    timescale_rand = np.random.choice(time_scale, size=num_samples, replace=True)

    if drop_last:
        number_of_batches = num_samples // batch_size  # Number of total batches with given data size
    else:
        number_of_batches = -(-num_samples // batch_size)

    if shuffle:
        p = np.random.permutation(num_samples)
    else:
        p = np.arange(num_samples)

    counter = 0
    while counter < number_of_batches:
        batch_index = p[batch_size * counter:min(num_samples, batch_size * (counter + 1))]
        batch_size = len(batch_index)
        coo = [[] for i in range(3)]

        sample_batch_idx = sample_index[batch_index % len(sample_index)]
        # timescale_batch = time_scale[batch_index // len(sample_index)]
        timescale_batch = timescale_rand[batch_index]
        # print(sample_batch_idx,timescale_batch)

        for bc in range(len(batch_index)):
            idx = sample_batch_idx[bc]
            ts = (np.round(times[idx]*timescale_batch[bc]/time_step).astype(np.int))
            us = units[idx]

            # Constrain spike length
            idx = (ts < nb_steps)
            ts = ts[idx]
            us = us[idx]
            # Add random spikes
            ts_r, us_r = poisson_spikes_gen(nb_units, nb_steps, rate, time_step, seed)
            ts = np.concatenate((ts, ts_r))
            us = np.concatenate((us, us_r))
            # Delete spikes at random with given prob.
            idx = poisson_spikes_indices(ts.size, p_del, seed)
            ts = ts[idx]
            us = us[idx]

            batch = [bc for _ in range(ts.size)]
            coo[0].extend(batch)
            coo[1].extend(ts.tolist())
            coo[2].extend(us.tolist())

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to_dense()
        y_batch = torch.tensor([class_list.index(a) for a in labels[sample_batch_idx].astype(np.int)], dtype=torch.long)

        X_batch[X_batch[:] > 1.] = 1.

        yield X_batch, y_batch, timescale_batch

        counter += 1

        def sparse_data_generator_scale(units, times, labels, prms, shuffle=True, epoch=0, drop_last=True):
            seed = prms['seed'] + epoch
            set_seed(seed)
            batch_size = prms['batch_size']
            nb_steps = prms['nb_steps']
            nb_units = prms['nb_inputs']
            time_step = prms['time_step']
            rate = prms['rate']
            p_del = prms['p_del']
            class_list = prms['class_list']
            time_scale = np.array(prms['time_scale'])

            sample_index = np.where(np.isin(labels, class_list))[0]  # Indices to access the data_paths

            # extend for multiple timescale
            num_samples = len(sample_index)
            # num_samples = len(sample_index)*len(time_scale)  # Number of samples in data

            # Draw a random timescale factor from the list
            timescale_rand = np.random.choice(time_scale, size=num_samples, replace=True)

            if drop_last:
                number_of_batches = num_samples // batch_size  # Number of total batches with given data size
            else:
                number_of_batches = -(-num_samples // batch_size)

            if shuffle:
                p = np.random.permutation(num_samples)
            else:
                p = np.arange(num_samples)

            counter = 0
            while counter < number_of_batches:
                batch_index = p[batch_size * counter:min(num_samples, batch_size * (counter + 1))]
                batch_size = len(batch_index)
                coo = [[] for i in range(3)]

                sample_batch_idx = sample_index[batch_index % len(sample_index)]
                # timescale_batch = time_scale[batch_index // len(sample_index)]
                timescale_batch = timescale_rand[batch_index]
                # print(sample_batch_idx,timescale_batch)

                for bc in range(len(batch_index)):
                    idx = sample_batch_idx[bc]
                    ts = (np.round(times[idx] * timescale_batch[bc] / time_step).astype(np.int))
                    us = units[idx]

                    # Constrain spike length
                    idx = (ts < nb_steps)
                    ts = ts[idx]
                    us = us[idx]
                    # Add random spikes
                    ts_r, us_r = poisson_spikes_gen(nb_units, nb_steps, rate, time_step, seed)
                    ts = np.concatenate((ts, ts_r))
                    us = np.concatenate((us, us_r))
                    # Delete spikes at random with given prob.
                    idx = poisson_spikes_indices(ts.size, p_del, seed)
                    ts = ts[idx]
                    us = us[idx]

                    batch = [bc for _ in range(ts.size)]
                    coo[0].extend(batch)
                    coo[1].extend(ts.tolist())
                    coo[2].extend(us.tolist())

                i = torch.LongTensor(coo)
                v = torch.FloatTensor(np.ones(len(coo[0])))

                X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to_dense()
                y_batch = torch.tensor([class_list.index(a) for a in labels[sample_batch_idx].astype(np.int)],
                                       dtype=torch.long)

                X_batch[X_batch[:] > 1.] = 1.

                yield X_batch, y_batch, timescale_batch

                counter += 1


def sparse_data_generator_normaldist(units, times, labels, prms, shuffle=True, epoch=0, drop_last=True):
    seed = prms['seed']+epoch
    set_seed(seed)
    batch_size = prms['batch_size']
    nb_steps = prms['nb_steps']
    nb_units = prms['nb_inputs']
    time_step = prms['time_step']
    rate = prms['rate']
    p_del = prms['p_del']
    class_list = prms['class_list']
    time_scale = np.array(prms['time_scale'])

    sample_index = np.where(np.isin(labels, class_list))[0]  # Indices to access the data_paths

    # extend for multiple timescale
    num_samples = len(sample_index)
    # num_samples = len(sample_index)*len(time_scale)  # Number of samples in data

    # Draw a random timescale factor from the list
    mu = 0
    sigma = 0.5
    timescale_rand = 2 ** np.random.normal(mu, sigma, size=num_samples)

    if drop_last:
        number_of_batches = num_samples // batch_size  # Number of total batches with given data size
    else:
        number_of_batches = -(-num_samples // batch_size)

    if shuffle:
        p = np.random.permutation(num_samples)
    else:
        p = np.arange(num_samples)

    counter = 0
    while counter < number_of_batches:
        batch_index = p[batch_size * counter:min(num_samples, batch_size * (counter + 1))]
        batch_size = len(batch_index)
        coo = [[] for i in range(3)]

        sample_batch_idx = sample_index[batch_index]
        timescale_batch = timescale_rand[batch_index]
        # print(sample_batch_idx,timescale_batch)

        for bc in range(len(batch_index)):
            idx = sample_batch_idx[bc]
            ts = (np.round(times[idx]*timescale_batch[bc]/time_step).astype(np.int))
            us = units[idx]

            # Constrain spike length
            idx = (ts < nb_steps)
            ts = ts[idx]
            us = us[idx]
            # Add random spikes
            ts_r, us_r = poisson_spikes_gen(nb_units, nb_steps, rate, time_step, seed)
            ts = np.concatenate((ts, ts_r))
            us = np.concatenate((us, us_r))
            # Delete spikes at random with given prob.
            idx = poisson_spikes_indices(ts.size, p_del, seed)
            ts = ts[idx]
            us = us[idx]

            batch = [bc for _ in range(ts.size)]
            coo[0].extend(batch)
            coo[1].extend(ts.tolist())
            coo[2].extend(us.tolist())

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to_dense()
        y_batch = torch.tensor([class_list.index(a) for a in labels[sample_batch_idx].astype(np.int)], dtype=torch.long)

        X_batch[X_batch[:] > 1.] = 1.

        yield X_batch, y_batch

        counter += 1