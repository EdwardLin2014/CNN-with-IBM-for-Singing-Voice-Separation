#########################################################################
## Step 0 - Import Library
import os, sys
import time
import numpy as np
import scipy.io as sio
import h5py

currTime = time.strftime("%Y%m%d_%H%M")
Tool_DNN_DirStr = '../../00_Tools/MyDNN-1.0/'
H5DirStr = '/home/users/sutd/1000791/scratch/NetInput_HDF5/iKala/'

CSOut = open('./Program_Output/trainAE_CSOut.txt','w+')
ModelDirStr = '/home/users/sutd/1000791/01_iKalaProject/01_CNN/model_' + currTime
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_DNN_DirStr))
import MyDNN as MD
import tensorflow as tf

numTFbins = 18441
numFrames = 9
numBins = 2049
numClass = 18441
TnumFrames = 323
hFrame = 107
TSolu = 9
FSolu = 15

MaxEpochs = 180
numTrain = 152
numValid = 50
numTest = 50
cost = np.zeros((MaxEpochs,3))
    
#########################################################################
## Step 0 - Obtain dataset
totaltic = time.time()

print('Start to obtain dataset. Please wait......')
H5FileName = H5DirStr + 'iKala_1024_4096_256_323_9_IBM_F32_FOrder.h5'

tic = time.time()
h5f = h5py.File(H5FileName, 'r')
trainSet = h5f['trainAE']
trainLabel = h5f['trainAE']
toc = time.time() - tic
print >>CSOut,'Obtained Training set need %.2f sec' % toc
CSOut.flush()

tic = time.time()
valSet = h5f['validAE']
valLabel = h5f['validAE']
toc = time.time() - tic
print >>CSOut,'Obtained validation set need %.2f sec' % toc
CSOut.flush()

tic = time.time()
testSet = h5f['testAE']
testLabel = h5f['testAE']
toc = time.time() - tic
print >>CSOut,'Obtained Test set need %.2f sec' % toc
CSOut.flush()

totaltoc = time.time() - totaltic
print >>CSOut,'Obtain all dataset needs %.2f sec' % totaltoc
CSOut.flush()

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
Conv1 = MD.conv2d_layer(x_image, 3,12,1,32,'Conv1')         # (3*12)*32 + 32 = 1,152 + 32 = 1,184
Conv2 = MD.conv2d_layer(Conv1,3,12,32,16,'Conv2')           # (3*12)*32*16 +16 = 18,432 + 16 = 18,448
MaxPool1 = MD.max_pool_1x12(Conv2,'MaxPool1')               # Image Size (9*171)
Conv3 = MD.conv2d_layer(MaxPool1,3,12,16,64,'Conv3')        # (3*12)*16*64 + 64 = 36,864 + 64 = 36,928 
Conv4 = MD.conv2d_layer(Conv3,3,12,64,32,'Conv4')           # (3*12)*64*32 + 32 = 73,728 + 32 = 73,760
MaxPool2 = MD.max_pool_1x12(Conv4,'MaxPool2')		    # Image Size (9*15)
# fully-connected layer + dropout
Conv4_flat = tf.reshape(MaxPool2, [-1, TSolu * FSolu * 32], name='Conv4_flat')   # 4,320
fc1_drop = tf.nn.dropout(Conv4_flat, keep_prob, name='drop1')
fc1 = MD.nn_layer(fc1_drop, TSolu * FSolu * 32, 2048, 'fullyConnect1')           # 4,320*2,048 + 2,048 = 8,847,360 + 2,048 = 8,849,408
fc2_drop = tf.nn.dropout(fc1, keep_prob, name='drop2')
fc2 = MD.nn_layer(fc2_drop, 2048, 512, 'fullyConnect2')                          # 2,048*512 + 512 = 1,048,576 + 512 = 1,049,088
# output layer
y = MD.nn_layer(fc2, 512, numClass, 'ouputLayer', act=tf.identity)               # 512*18,441 + 18,441 = 9,441,792 + 18,441 = 9,460,233
# Total Trainable Parameters = 19,489,049
# Define loss and optimizer
crossEntropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y, name='crossEntropy'), name='lossValue')
train_step = tf.train.AdamOptimizer(1e-4, name='adam').minimize(crossEntropy)
tf.global_variables_initializer().run()
toc = time.time() - tic
print >>CSOut,'Create CNN Model needs %.2f sec at %s' % (toc,time.strftime("%Y%m%d_%H%M"))
CSOut.flush()
#print('Total Parameters')
#MD.getTotalParameters()

#########################################################################
## Step 2 - Train the CNN and Write the Log
totalic = time.time()
saver = tf.train.Saver()
minValCost = 1
for i in range(MaxEpochs):
    # Training
    tic = time.time()
    for t in range(numTrain):
        StartIdx = t*TnumFrames
        EndIdx = StartIdx + hFrame
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 0.5})
        #print('1 - %dth Training'%t)
        StartIdx = EndIdx
        EndIdx = StartIdx + hFrame
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 0.5})
        #print('2 - %dth Training'%t)
        StartIdx = EndIdx
        EndIdx = (t+1)*TnumFrames
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx,:], keep_prob: 0.5})
        #print('3 - %dth Training'%t)
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
        #print('%d: cost[%d,0] = %'%{t,i,cost[i,0]})
    cost[i,0] /= numTrain
    toc = time.time() - tic
    print >>CSOut,'%dth Train epoch; Cost = %.8f; times need %.2f sec at %s' % (i,cost[i,0],toc,time.strftime("%Y%m%d_%H%M"))

    CSOut.flush()

    # Calculate Loss Value for Validation set for 1 epoch
    tic = time.time()
    for t in range(numValid):
        StartIdx = t*TnumFrames
        EndIdx = StartIdx + hFrame
        tmpCost1 = sess.run(crossEntropy, feed_dict={x:valSet[StartIdx:EndIdx,:], y_:valLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = StartIdx + hFrame
        tmpCost2 = sess.run(crossEntropy, feed_dict={x:valSet[StartIdx:EndIdx,:], y_:valLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        StartIdx = EndIdx
        EndIdx = (t+1)*TnumFrames
        tmpCost3 = sess.run(crossEntropy, feed_dict={x:valSet[StartIdx:EndIdx,:], y_:valLabel[StartIdx:EndIdx,:], keep_prob: 1.0})
        cost[i,1] += (tmpCost1+tmpCost2+tmpCost3)/3
        #print('%d: cost[%d,1] = %'%{t,i,cost[i,1]})
    cost[i,1] /= numValid
    if cost[i,1] < minValCost:
    ## Step 3 - Save the Trained Model
        stic = time.time()
        if tf.gfile.Exists(ModelDirStr):
            tf.gfile.DeleteRecursively(ModelDirStr)
        tf.gfile.MakeDirs(ModelDirStr)
        saver.save(sess, ModelDirStr+'/model', global_step=i)
        minValCost = cost[i,1]
        stoc = time.time() - stic
        print >>CSOut,'Save the %dth Trained Model needs %.2f sec at %s' % (i,stoc,time.strftime("%Y%m%d_%H%M"))
        CSOut.flush()
    toc = time.time() - tic
    print >>CSOut,'%dth Valid epoch; Cost = %.8f; times need %.2f sec at %s' % (i,cost[i,1],toc,time.strftime("%Y%m%d_%H%M"))
    CSOut.flush()
    
    # Calculate Loss Value for Test set for 1 epoch
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
        #print('%d: cost[%d,2] = %'%{t,i,cost[i,2]})
    cost[i,2] /= numTest
    toc = time.time() - tic
    print >>CSOut,'%dth Test epoch; Cost = %.8f; times need %.2f sec at %s' % (i,cost[i,2],toc,time.strftime("%Y%m%d_%H%M"))

    print >>CSOut,'-------------------------------------------------'
    CSOut.flush()

totaloc = time.time() - totalic
print >>CSOut,'Total Training Time needs %.2f sec' % totaloc
CSOut.flush()

#########################################################################
## Step 4 - Save the loss function for ploting
tic = time.time()
sio.savemat('cost_'+currTime+'.mat',mdict={'cost': cost})
saver.save(sess, './AtStep180/model', global_step=i)
toc = time.time() - tic
print >>CSOut,'Save the Trained Model needs %.2f sec' % toc
CSOut.flush()
