# CNN-with-IBM-for-Singing-Voice-Separation

Please first download the trained CNN from the following site
https://www.dropbox.com/sh/1p521wsk01buean/AAAx4S1uT7ToZP3rwqK-bbjIa?dl=0
Then put them under 01_iKalaProject Folder

[model_20170930_1142]
The model is trained in the first 180 epochs at the parameters initialization steps
[model_20171001_1105]
The model is trained in the next 120 epochs at the parameters initialization steps
[model_20171002_0029]
The model is trained in the first 180 epochs at the actual training
[model_20171002_1756]
The model is trained in the next 120 epochs at the actual training

[model usage]
Before using model, change the path info in "checkpoint" file to match the full path of "mode_yyyymmdd_HHMM"

[iKala Dataset]
Put all the files, which is under the Wavfile of the ikala dataset, to "Wavfile" folder.
iKala dataset can be obtained from the following site
http://mac.citi.sinica.edu.tw/ikala/

[Reproduce Paper Result]
Simply execute the python files one by one, based on their sequency number 
e.g. "00_<filename>","01_<filename>", so on and so forth

