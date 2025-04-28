Code for the publication "PanThera: Predicting the effect of Combination Therapy using Deep Neural Networks".  

# Overview
PanThera aims at predicting arbitrary combination therapies. A pre-trained ensemble containing 10 replicas of PanThera trained on different subsets of the ALMANAC data, and an interface to generate predictions for arbitrary treatments are provided.  

# Requirements
This package is supported for Windows, macOS and Linux. The package has been tested using Debian GNU/Linux 10 and Ubuntu 20. GPU acceleration with CUDA 12.4 support is required to run our experiments.  
Python 3.7.3 and 3.9 were used, and the required packages to run the code are:  

```
numpy==1.24.4
optuna==3.1.0
pandas==1.5.3
rdkit==2022.3.5
scikit-learn==1.3.2 
scipy==1.10.1
synergy==1.0.0
torch==2.4.0
torchmetrics==0.10.0
```

# Installation 
For the usage of our package, no particular Python package is provided. For the installation of Python on different operating systems the instructions can be found at https://www.python.org/downloads/. Once Python is installed the required libraries can be installed using the command `pip install -r requirements.txt`. The installation time will greatly depend on the time needed to download and install PyTorch with CUDA support, which could take up to one hour depending on the network being used.  

# Generating predictions for Combination Therapies
Predictions can be performed using the command `python3 predict.py --file {your_file.csv} --cuda {your_cuda_device_number}`. `your_file.csv` must contains columns `[SMILES_1, ..., SMILES_N, CONC1	,...,	CONCN, CELL_NAME]`. Where SMILES_X, CONCX contains the canonical smiles and concentration of a drug present in the combination therapy, and CELL_NAME contains the name of one of the cell-lines of the NCI60 where the combination therapy will be tested. Each row corresponds to one combination therapy.  

The pre-trained models can be downloaded and extracted from [Zenodo](/guides/content/editing-an-existing-page#modifying-front-matter](https://zenodo.org/records/14216168)  

The output will be stored in {*your_file_prediction.csv*}, and an additional column, `prediction`, will contain the average predicted inhibitory effect of the ensemble.  

Additionally, it can be ran as a docker image:  
first the image is built:  

```docker build -f Dockerfile.base -t panthera_base .```

```docker build -f Dockerfile.predict -t panthera_predict .```

Then predictions can be generated for the file `example_quadruplet.csv` using CUDA device number 0 by executing:

```docker run --runtime=nvidia --gpus all -v $(pwd)/results:/app/results -e FILE_NAME=example_quadruplet.csv -e CUDA=0 panthera_predict ```

where FILE_NAME denotes the csv file for which the predictions will be generated and CUDA denotes the CUDA device number.  

# Software Demo

```python3 predict.py --file example_quadruplet.csv --cuda 0```  
will generate predictions for 2 combination therapies consisting of 4 drugs at different concentrations. The output will be stored in `example_quadruplet_prediction.csv`. The runtime will depend greatly on the characteristics of your system but should be under 1 minute. 

# Training the model from scratch
The data and optimal hyperparameters can be downloaded from [Zenodo](/guides/content/editing-an-existing-page#modifying-front-matter](https://zenodo.org/records/14216168) and must be extracted in the app folder.  

Then the model can be trained using  

```python3 train_synergy.py --fold [FOLD_NUMBER] --cuda [CUDA_NUMBER] --setting [SETTING] --data_path [FILE.CSV] --hyperparameter_study [STUDY_NAME]```

where each argument denotes:  

[FOLD_NUMBER]: The index of the split used as training data for the model  
[CUDA_NUMBER]: The index of the cuda device   
[SETTING]: The setting used for splitting the data into training and testing  
[FILE.CSV]: An optional file used to train the model, if left blank, it will use the ALMANAC  
[STUDY_NAME]: An optional Optuna study name used for selecting a model configuration. If left blank, it will used the optimal hyperparameters obtained from our hyperaparameter study.  


Alternatively, we provide the pipeline for training the model as a contained:  

```docker build -f Dockerfile.base -t panthera_base .```  

```docker build -f Dockerfile.train -t panthera_train .```  

```docker run --runtime=nvidia --gpus all -v $(pwd)/results:/app/results -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data -e FOLD=0 -e CUDA=0 -e SETTING=drug_combination_discovery panthera_train ```  

where FOLD is the fold number, CUDA denotes the CUDA device number, SETTING is the split setting used to train the model.  
Additionally, one can add -e DATA_PATH=yourfile.csv to train the model using a custom file  

# Finding optimal hyperparameters
We provide the optimal hyperparameters used in our experiments, and our Bayesian optimization-based pipeline was used to find them  

This is done by calling 
```python3 optimize_model.py --cuda [CUDA_NUMBER] --setting [SETTING] --data_path [FILE.CSV]```  


where each argument denotes:  
[CUDA_NUMBER]: The index of the cuda device  
[SETTING]: The setting used for splitting the data into training and testing  
[FILE.CSV]: An optional file used to train the model, if left blank, it will use the ALMANAC  

Alternatively, we provide the hyperparameters optimization pipeline in a separate docker container:  

```docker build -f Dockerfile.base -t panthera_base .```  

```docker build -f Dockerfile.optimize_hyperparameters -t panthera_optimize .```  

```docker run --runtime=nvidia --gpus all  -v $(pwd)/studies:/app/studies -v $(pwd)/data:/app/data -e CUDA=0 panthera_optimize ```  


CUDA denotes the CUDA device number  
Additionally one can add -e SETTING=other_setting to train the model using a setting that's not the prediction of drug combinations -e DATA_PATH=yourfile.csv to train the model using a custom file  

The output will be stored in studies/{setting}_new, and {setting}_new can be passed as an optional argument to train the model using the selected hyperparameters.   
Note that this can take very long; several instances can be run in parallel, but training a single instance can take up to 2 days of computation.   

# Troubleshooting
For running the models in containers, a CUDA-capable machine with Docker (https://docs.docker.com/engine/install/) is required.
If the  NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is not installed before building the images, it can lead to errors where the GPU is not detected.  
Furthermore, it requires the CUDA toolkit at version 12.4+ (https://developer.nvidia.com/cuda-12-4-0-download-archive)  
