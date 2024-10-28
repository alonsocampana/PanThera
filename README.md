Code for the publication "PanThera: Predicting the effect of Combination Therapy using Deep Neural Networks".

# Overview
PanThera aims at predicting arbitrary combination therapies. A pre-trained ensemble containing 10 replicas of PanTHera trained on different subsets of the ALMANAC data, and an interface to generate predictions for arbitrary treatments are provided. 

# Requirements
This package is supported for Windows, macOS and Linux. The package has been tested using Debian GNU/Linux 10. GPU acceleration with CUDA 11.8 support is required to run our experiments.
Python 3.7.3 was used, and the required packages to run the code are:
```
numpy==1.26.2 
optuna==3.1.0
pandas==1.5.3
rdkit==2022.3.5
scikit-learn==1.3.2 
scipy==1.11.4 
synergy==0.5.1
torch==2.1.1+cu118
torchmetrics==0.10.0
```
# Installation 
For the usage of our package, no particular python package is provided. For the installation of python on different operating systems the instructions can be found in https://www.python.org/downloads/. Once python is installed the required libraries can be installed using the command `pip install -r requirements.txt`.

# Generating predictions for Combination Therapies

Predictions can be performed using the command `python3 predict.py --file {your_file.csv} --cuda {your_cuda_device_number}`. `your_file.csv` must contains columns `[SMILES_1, ..., SMILES_N, CONC1	,...,	CONCN, CELL_NAME]`. Where SMILES_X, CONC_X contains the cannonical smiles and concentration of a drug present in the combination therapy, and CELL_NAME contains the name of one of the cell-lines of the NCI60 where the combination therapy will be tested. Each row corresponds to one combination therapy.
The output will be stored in {your_file_prediction.csv}, and an additional column, `prediction`, will contain the average predicted inhibitory effect of the ensemble. 

# Software Demo

`python3 predict.py --file example_quadruplet.csv --cuda 0` will generate predictions for 2 combination therapies consisting of 4 drugs at different concentrations. The output will be stored in `example_quadruplet_prediction.csv`. The runtime will depend greatly on the characteristics of your system but should be under 1 minute. 
