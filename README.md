# Code for  Neural heterogeneity promotes robust learning

[![DOI](https://zenodo.org/badge/328637490.svg)](https://zenodo.org/badge/latestdoi/328637490)

## Requirements

For surrogate gradient descent based methods:
* [pyTorch >=1.3](https://pytorch.org/)
* [scipy](https://www.scipy.org/) 
* [matplotlib](https://matplotlib.org/stable/users/installing.html) 
* [seaborn](https://seaborn.pydata.org/) 
* [tqdm](https://tqdm.github.io/) 
* [dill](https://dill.readthedocs.io/en/latest/dill.html) 
* [pytables](https://www.pytables.org/usersguide/installation.html) (for reading .h5 files for SHD and N-MNIST datasets)

For FORCE training based methods:
* [MATLAB](https://www.mathworks.com/products/matlab.html)

For plotting experimentally observed time constant distributions:
* [matplotlib](https://matplotlib.org/stable/users/installing.html) 
* [scipy](https://www.scipy.org/) 
* [allensdk](https://allensdk.readthedocs.io/en/latest/install.html)

## Running Surrogate based experiments:

In order to run an experiments you simply need to navigate to `SuGD_code` and run

```
python main.py
```
This will run with the default parameters, which were set for SHD dataset. You can specify them, for instance to initialise the network with heterogeneous time constants following a uniform distribution and training the time constants:

```
python main.py --het_ab 1 --dist_prms uniform --train_ab 1
```

A comprehensive list of parameters and a description is provided within `main.py`.

All datasets can be downloaded from the links provided below. Running the N-MNIST and DVS dataset require an extra step to convert the data into .h5 format, This can be done by navegating to `SuGD_code/NMNIST` or `SuGD_code/DVS` and running `save_data_nmnist.py` or `save_data_dvs.py` respectively. You can specify the original dataset directory with `--data_dir` and where you want to save the converted dataset `--output_filename` arguments.

## Running FORCE based experiments:

Navigate to `FORCE_code`. This directory contains the script `LIFSONGBIRD_HET.m`. This script runs FORCE in the three regimes specified in the paper and then plots the results. You may want to comment the last section which runs the gridsearch as it takes very long.

## Plotting experimentally observed time constants (Allen and Manis datasets):

Go to `Experiment_taudist_code`. Since it is problematic to access the time constants in the Paul Manis dataset, you can first run `manis_data.py` to extract and save them in a text file. You may then run `time_constant_distributions.py` to plot the distributions of Paul Manis dataset, as well as Allen Atlas dataset. The plots will also include their corresponding Gamma and Log-normal fits.

## Datasets:

The following links will direct you to the place where you can download all datasets/data:

* [Heidelberg Spiking Datasets (SHD and SSC)](https://compneuro.net/posts/2019-spiking-heidelberg-digits/)

* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

* [Neuromorphic-MNIST](https://www.garrickorchard.com/datasets/n-mnist)

* [DVS Gesture](https://www.research.ibm.com/dvsgesture/)

* [Zebra Finch bird call](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3192758/bin/pone.0025506.s002.wav)

* [Allen Atlas](https://allensdk.readthedocs.io/en/latest/ )

* [Paul Manis dataset](https://figshare.com/articles/dataset/Raw_voltage_and_current_traces_for_current-voltage_IV_relationships_for_cochlear_nucleus_neurons_/8854352 )


