# neural_heterogeneity

## Datasets:
Heidelberg Spiking Datasets (SHD and SSC):
https://compneuro.net/posts/2019-spiking-heidelberg-digits/ 

Fashion-MNIST: 
https://github.com/zalandoresearch/fashion-mnist 

N-MNIST:
https://www.garrickorchard.com/datasets/n-mnist 

DVS Gesture:
https://www.research.ibm.com/dvsgesture/ 

Zebra Finch bird song:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3192758/bin/pone.0025506.s002.wav 58/

Allen Atlas:
https://allensdk.readthedocs.io/en/latest/ 

Paul Manis dataset:
https://figshare.com/articles/dataset/Raw_voltage_and_current_traces_for_current-voltage_IV_relationships_for_cochlear_nucleus_neurons_/8854352 


## Usage

The file format used in the codes are .h5 files, based on the original format of SHD and SSC datasets. 

For other datasets such as DVS and N-MNIST, To convert the datasets from their original format to .h5, there is a script under the sub-folder for each dataset. Here gives the example for each dataset

DVS:
```
python save_data_dvs.py --data_dir 'DVS Gesture dataset/DvsGesture/' --input_filename 'trials_to_train.txt' --output_filename "train.h5"
```

N-MNIST:
```
python save_data_nmnist.py --data_dir 'N-MNIST/Train/' --output_filename "train.h5"
```

To train the network, you can simply run main.py with specified options, such as
```
python main.py --dataset DVS --nb_inputs 32768
```

After training the models, you can run results.py to compare the results from different heterogeneity settings.