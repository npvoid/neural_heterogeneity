from loadaerdat import loadaerdata
import csv
from tables import *
from numpy import *
import argparse
from os.path import join as pjoin


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Data Directory')
parser.add_argument('--input_filename', type=str,  help='Input containing the list of data')
parser.add_argument('--output_filename', type=str, help='Output File Name')
parser.add_argument('--class_list', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

prms = vars(parser.parse_args())

labels = []

h5file = open_file(prms['output_filename'], mode="w", title="DVS training dataset")
spikes = h5file.create_group(h5file.root, "spikes", "Spikes in eventbased form")
units = h5file.create_vlarray(spikes, 'units', Int32Atom(shape=()), "Spike neuron IDs, int")
times = h5file.create_vlarray(spikes, 'times', Float32Atom(shape=()), "Spike Times (in seconds), float")

data_count = 0
with open(pjoin(prms['data_dir'], prms['input_filename']), "r") as f:
    for line in f:
        data_filename = pjoin(prms['data_dir'],line[:-7])
        timestamps, xaddr, yaddr, pol = loadaerdata(data_filename+".aedat", 0, 0)
        with open(data_filename+'_labels.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0] in str(prms['class_list']):
                    startTs = int(row[1])
                    endTs = int(row[2])
                    idx = (timestamps >= startTs) & (timestamps < endTs)
                    timestamps_class = (timestamps[idx] - startTs)*1e-6
                    print(timestamps_class.max())
                    times.append(array(timestamps_class))
                    units.append(array(xaddr[idx]+yaddr[idx]*128+pol[idx]*128*128))
                    labels.append(int(row[0]))
                    data_count = data_count + 1

shape = (data_count, )
labels_array = h5file.create_carray(h5file.root, 'labels', Int32Atom(), shape=shape,  title="Data Labels")
labels_array[:] = array(labels)

h5file.close()
