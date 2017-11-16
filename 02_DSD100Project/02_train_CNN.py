#########################################################################
## Step 0 - Import Library
import os, sys
import time
import numpy as np
import scipy.io as sio
import h5py

currTime = time.strftime("%Y%m%d_%H%M")
Tool_DNN_DirStr = '../../00_Tools/TFWrapper-1.0/'           # Tensorflow Wrapper
H5DirStr = '../00_HDF5/'                                    # CNN input
OldModelDirStr = './model_20171004_0203'
NewModelDirStr = './model_' + currTime
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_DNN_DirStr))
import tensorflow as tf

numTFbins = 18441                   # 2049(0~11.025kHz) * 9 (~104.49 msec)
numFrames = 9                       # Each input for CNN is ~104.49 msec
numBins = 2049                      # Analysis DFT Size, 185.76ms
numClass = 18441                    # Each F-T bin is treated as a class
TnumFrames = 323                    # 323 CNN input instances per one song
hFrame = 107                        # number of instances for each batch update
TSolu = 9                           # Image Size (Time) after final pooling
FSolu = 15                          # Image Size (Frequency) after final pooling

MaxEpochs = 300                     # Maximum number of epochs
numTrain = 347                      # number of training clips, number of instance: 112,081
numValid = 347                      # number of validation clips, number of instance: 112,081
numTest = 298                       # number of test clips, number of instance: 96,254
cost = np.zeros((MaxEpochs,3))      # loss function for each dataset
    
#########################################################################
## Step 0 - Obtain dataset
totaltic = time.time()

print('Start to obtain dataset. Please wait......')
H5FileName = H5DirStr + 'DSD100_IBM.h5'

tic = time.time()
h5f = h5py.File(H5FileName, 'r')
trainSet = h5f['train']                                   # Mixture Spectrogram
trainLabel = h5f['trainLabel']                            # Vocal IBM Label
toc = time.time() - tic
print('Obtained Training set need %.2f sec' % toc)

tic = time.time()
validSet = h5f['valid']                                   # Mixture Spectrogram
validLabel = h5f['validLabel']                            # Vocal IBM Label
toc = time.time() - tic
print('Obtained Valid set need %.2f sec' % toc)

tic = time.time()
testSet = h5f['test']                                     # Mixture Spectrogram
testLabel = h5f['testLabel']                              # Vocal IBM Label
toc = time.time() - tic
print('Obtained Test set need %.2f sec' % toc)

totaltoc = time.time() - totaltic
print('Obtain all dataset needs %.2f sec' % totaltoc)

#########################################################################
## Step 1 - Restore the model
tic = time.time()
sess = tf.Session() 
saver = tf.train.import_meta_graph(OldModelDirStr+'/model-292.meta')
saver.restore(sess,tf.train.latest_checkpoint(OldModelDirStr))
# Now, let's access and create placeholders variables
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("xInput:0")
y_ = graph.get_tensor_by_name("yInput:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
crossEntropy = graph.get_tensor_by_name("lossValue:0")
train_step = graph.get_operation_by_name("adam")
toc = time.time() - tic
print('Restore the model needs %.2f sec at %s' % (toc,time.strftime("%Y%m%d_%H%M")))

#########################################################################
## Step 2 - Train the CNN and Write the Log
totalic = time.time()
saver = tf.train.Saver()
minValidCost = 1
for i in range(MaxEpochs):
    # Training
    tic = time.time()
    for t in range(numTrain):
        StartIdx = t*TnumFrames
        EndIdx = StartIdx + hFrame
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 0.5})
        StartIdx = EndIdx
        EndIdx = StartIdx + hFrame
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 0.5})
        StartIdx = EndIdx
        EndIdx = (t+1)*TnumFrames
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 0.5})
    for t in range(numTrain):
        StartIdx = t*TnumFrames
        EndIdx = StartIdx + hFrame
        tmpCost1 = sess.run(crossEntropy, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = StartIdx + hFrame
        tmpCost2 = sess.run(crossEntropy, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = (t+1)*TnumFrames
        tmpCost3 = sess.run(crossEntropy, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        cost[i,0] += (tmpCost1+tmpCost2+tmpCost3)/3
    cost[i,0] /= numTrain
    toc = time.time() - tic
    print('%dth Train epoch; Cost = %.8f; times need %.2f sec at %s' % (i,cost[i,0],toc,time.strftime("%Y%m%d_%H%M")))

    # Calculate Loss Value for Validation set for 1 epoch
    tic = time.time()
    for t in range(numValid):
        StartIdx = t*TnumFrames
        EndIdx = StartIdx + hFrame
        tmpCost1 = sess.run(crossEntropy, feed_dict={x:validSet[StartIdx:EndIdx,:], y_:validLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = StartIdx + hFrame
        tmpCost2 = sess.run(crossEntropy, feed_dict={x:validSet[StartIdx:EndIdx,:], y_:validLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = (t+1)*TnumFrames
        tmpCost3 = sess.run(crossEntropy, feed_dict={x:validSet[StartIdx:EndIdx,:], y_:validLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        cost[i,1] += (tmpCost1+tmpCost2+tmpCost3)/3
    cost[i,1] /= numValid
    if cost[i,1] < minValidCost:
    ## Step 3 - Save the Trained Model
        stic = time.time()
        if tf.gfile.Exists(NewModelDirStr):
            tf.gfile.DeleteRecursively(NewModelDirStr)
        tf.gfile.MakeDirs(NewModelDirStr)
        saver.save(sess, NewModelDirStr+'/model', global_step=i)
        minValidCost = cost[i,1]
        stoc = time.time() - stic
        print('Save the %dth Trained Model needs %.2f sec at %s' % (i,stoc,time.strftime("%Y%m%d_%H%M")))
    toc = time.time() - tic
    print('%dth Valid epoch; Cost = %.8f; times need %.2f sec at %s' % (i,cost[i,1],toc,time.strftime("%Y%m%d_%H%M")))
    
    # Testing
    tic = time.time()
    for t in range(numTest):
        StartIdx = t*TnumFrames
        EndIdx = StartIdx + hFrame
        tmpCost1 = sess.run(crossEntropy, feed_dict={x:testSet[StartIdx:EndIdx,:], y_:testLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = StartIdx + hFrame
        tmpCost2 = sess.run(crossEntropy, feed_dict={x:testSet[StartIdx:EndIdx,:], y_:testLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = (t+1)*TnumFrames
        tmpCost3 = sess.run(crossEntropy, feed_dict={x:testSet[StartIdx:EndIdx,:], y_:testLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        cost[i,2] += (tmpCost1+tmpCost2+tmpCost3)/3
    cost[i,2] /= numTest
    toc = time.time() - tic
    print('%dth Test epoch; Cost = %.8f; times need %.2f sec at %s' % (i,cost[i,2],toc,time.strftime("%Y%m%d_%H%M")))
    print('-------------------------------------------------')

totaloc = time.time() - totalic
print('Total Training Time needs %.2f sec' % totaloc)

#########################################################################
## Step 4 - Save the loss function for ploting
tic = time.time()
sio.savemat('cost_'+currTime+'.mat',mdict={'cost': cost})
toc = time.time() - tic
print('Save the Trained Model needs %.2f sec' % toc)
