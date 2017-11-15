# CNN-with-IBM-for-Singing-Voice-Separation

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
Put all the files, which is under the Wavfile of the ikala dataset, to "Wavfile" folder.
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

## 02 DSD100 Dataset


