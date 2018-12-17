# CNN-with-IBM-for-Singing-Voice-Separation

## Getting Started
This is the source code to reproduce the results stated at my journal paper:
Kin Wah Edward Lin, Balamurali B T, Enyan Koh, Simon Lui and Dorien Herremans. (2018)
"Singing Voice Separation using a Deep Convolutional Neural Network Trained by Ideal Binary Mask and Cross Entropy"
Special Issue on Deep Learning for Music and Audio in Springer’s Neural Computing and Applications
Https://doi.org/10.1007/s00521-018-3933-z

## Prerequisites
* The source code must be executed using GPU with at least 12GB memory , e.g. NVIDIA GeForce GTX TITAN X. 
* Install the following python packages.
```
pip install resampy
pip install dsdtools
```

## 01 iKala Dataset
* Please first download the trained CNN from the following site
https://www.dropbox.com/sh/1p521wsk01buean/AAAx4S1uT7ToZP3rwqK-bbjIa?dl=0
* Then put them under 01_iKalaProject Folder
* model_20170930_1142 is trained in the first 180 epochs at the parameters initialization steps
* model_20171001_1105 is trained in the next 120 epochs at the parameters initialization steps
* model_20171002_0029 is trained in the first 180 epochs at the actual training
* model_20171002_1756 is trained in the next 120 epochs at the actual training

### Model usage
Before using model, change the path info in "checkpoint" file to match the full path of "mode_yyyymmdd_HHMM"

### Dataset
Put all the files, which is under the Wavfile of the ikala dataset, to "01_iKalaProject/Wavfile" folder.
iKala dataset can be obtained from the following site http://mac.citi.sinica.edu.tw/ikala/

### Reproduce Paper Result
Simply execute the python files one by one, based on their sequency number 
```
python 01_iKalaProject/00_prePro_HDF5.py
python 01_iKalaProject/01_parmInit_AE.py
python 01_iKalaProject/02_train_CNN.py
python 01_iKalaProject/03_postPro_Eval.py
python 01_iKalaProject/04_BSS_Eval.py
```
### Paper Result
The excel file, which contains all results of each clip, can be found at
https://www.dropbox.com/s/a9qumoobxm6a4u9/CNN_035.xlsx?dl=0

## 02 DSD100 Dataset
* Please first download the trained CNN from the following site
https://www.dropbox.com/sh/uln8f0429aksl0a/AAD9pJBnn5usn06qC3JagrfKa?dl=0
* Then put them under 01_iKalaProject Folder
* model_20171004_0203 is trained with 300 epochs at the parameters initialization steps
* model_20171006_0945 is trained with 300 epochs at the actual training

### Model usage
Before using model, change the path info in "checkpoint" file to match the full path of "mode_yyyymmdd_HHMM"

### Dataset
Put "Mixtures" and "Sources" Folders of the DSD100 dataset, to "02_DSD100Project/Wavfile" folder.
DSD100 dataset can be obtained from the following site https://www.sisec17.audiolabs-erlangen.de/

### Reproduce Paper Result
Simply execute the python files one by one, based on their sequency number 
```
python 02_DSD100Project/00_prePro_HDF5.py
python 02_DSD100Project/01_parmInit_AE.py
python 02_DSD100Project/02_train_CNN.py
python 02_DSD100Project/03_postPro_Eval.py
python 02_DSD100Project/04_BSS_Eval.py
```

### Paper Result
The excel file, which contains all results of each clip, can be found at
https://www.dropbox.com/s/ijvpusrdp7lxu3l/CNN_015.mat?dl=0
