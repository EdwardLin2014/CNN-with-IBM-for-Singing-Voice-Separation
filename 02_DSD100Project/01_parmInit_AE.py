#########################################################################
## Step 0 - Import Library
import os, sys
import time
import numpy as np
import scipy.io as sio
import h5py

currTime = time.strftime("%Y%m%d_%H%M")
Tool_DNN_DirStr = '../00_Tools/TFWrapper-1.0/'              # Tensorflow Wrapper
H5DirStr = '../00_HDF5/'                                    # CNN input
ModelDirStr = './model_' + currTime
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_DNN_DirStr))
import TFWrapper as TFW
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
trainSet = h5f['trainAE']                           # Vocal Spectrogram extracted from IBM x mX
trainLabel = h5f['trainAE']                         # Vocal Spectrogram extracted from IBM x mX
toc = time.time() - tic
print('Obtained Training set need %.2f sec' % toc)

tic = time.time()
validSet = h5f['validAE']                           # Vocal Spectrogram extracted from IBM x mX
validLabel = h5f['validAE']                         # Vocal Spectrogram extracted from IBM x mX
toc = time.time() - tic
print('Obtained Valid set need %.2f sec' % toc)

tic = time.time()
testSet = h5f['testAE']                             # Vocal Spectrogram extracted from IBM x mX
testLabel = h5f['testAE']                           # Vocal Spectrogram extracted from IBM x mX
toc = time.time() - tic
print('Obtained Test set need %.2f sec' % toc)

totaltoc = time.time() - totaltic
print('Obtain all dataset needs %.2f sec' % totaltoc)

#########################################################################
## Step 1 - Create ConvNet
tic = time.time()
sess = tf.InteractiveSession()
# Input and Ouput Layer & placeholder
x = tf.placeholder(tf.float32, [None, numTFbins], name='xInput')
y_ = tf.placeholder(tf.float32, [None, numClass], name='yInput')
x_image = tf.reshape(x, [-1,numFrames,numBins,1], name='x_image')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# conv & pool layer - activation='relu' - tf.contrib.layers.xavier_initializer 
Conv1 = TFW.conv2d_layer(x_image, 3,12,1,32,'Conv1')         # (3*12)*32 + 32 = 1,152 + 32 = 1,184
Conv2 = TFW.conv2d_layer(Conv1,3,12,32,16,'Conv2')           # (3*12)*32*16 +16 = 18,432 + 16 = 18,448
MaxPool1 = TFW.max_pool_1x12(Conv2,'MaxPool1')
Conv3 = TFW.conv2d_layer(MaxPool1,3,12,16,64,'Conv3')        # (3*12)*16*64 + 64 = 36,864 + 64 = 36,928 
Conv4 = TFW.conv2d_layer(Conv3,3,12,64,32,'Conv4')           # (3*12)*64*32 + 32 = 73,728 + 32 = 73,760
MaxPool2 = TFW.max_pool_1x12(Conv4,'MaxPool2')
# fully-connected layer + dropout
Conv4_flat = tf.reshape(MaxPool2, [-1, TSolu * FSolu * 32], name='Conv4_flat')   # 4320
fc1_drop = tf.nn.dropout(Conv4_flat, keep_prob, name='drop1')
fc1 = TFW.nn_layer(fc1_drop, TSolu * FSolu * 32, 2048, 'fullyConnect1')           # 4320*2048 + 2048 = 8,847,360 + 2048 = 8,849,408
fc2_drop = tf.nn.dropout(fc1, keep_prob, name='drop2')
fc2 = TFW.nn_layer(fc2_drop, 2048, 512, 'fullyConnect2')                          # 2048*512 + 512 = 1,048,576 + 512 = 1,049,088
# output layer
y = TFW.nn_layer(fc2, 512, numClass, 'ouputLayer', act=tf.identity)               # 512*18441 + 18441 = 9,441,792 + 18,441 = 9,460,233
# Total Trainable Parameters = 19,489,049
# Define loss and optimizer
crossEntropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y, name='crossEntropy'), name='lossValue')
train_step = tf.train.AdamOptimizer(1e-4, name='adam').minimize(crossEntropy)
tf.global_variables_initializer().run()
toc = time.time() - tic
print('Create CNN Model needs %.2f sec at %s' % (toc,time.strftime("%Y%m%d_%H%M")))

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
        if tf.gfile.Exists(ModelDirStr):
            tf.gfile.DeleteRecursively(ModelDirStr)
        tf.gfile.MakeDirs(ModelDirStr)
        saver.save(sess, ModelDirStr+'/model', global_step=i)
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
