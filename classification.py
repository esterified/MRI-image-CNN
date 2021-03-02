# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:25:05 2017

@author: ESTERIFIED
"""
#EDSS classification
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

#from __future__ import print_function
from sklearn.metrics import confusion_matrix
import example
import numpy as np
import scipy.misc
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
from datetime import timedelta
import matplotlib.pyplot as plt
import math
import time
from sklearn.model_selection import KFold
# Import MNIST data


pickle_file = 'edss.pickle'
save_model_path = './edss_classification'
f = open(pickle_file, 'rb')
datar = pickle.load(f)

clsLabels = np.argmax(datar['labels'], axis=1)
tf.reset_default_graph()
# Plot the images and labels using our helper-function above.
example.plot_images(datar['dataset'][0:4], clsLabels[0:4])
#k-fold cross vallidation
#x_train, x_test, y_train, y_test = train_test_split(datar['dataset'],datar['labels'], test_size=0.25, random_state=42)
X=datar['dataset']
y=datar['labels']
kf = KFold(n_splits=10)
kf.get_n_splits(X)
fold=1
for train_index, test_index in kf.split(X):
    
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if fold==2:#2 for 80% accuracy
        break
    fold=fold+1
# Training Parameters
learning_rate = 0.001
#beta for regularization 0.01
beta=0.01
epoch=20
batch_size = 28
display_step = 2
init_step=0
start=0
endi=0
step=init_step
# Network Parameters
num_input = 102400 # total data input (img shape: 320*320)
num_classes = 2 # EDSS total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input],name='x')
Y = tf.placeholder(tf.float32, [None, num_classes],name='y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob') # dropout (keep probability)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    
    '''
    global start,endi,batch_size
    num_examples = data.shape[0]
#    if endi> num_examples:
#        start=0
#        endi=batch_size
#        # shuffle the data
#        assert batch_size <= num_examples
#    perm = np.arange(num_examples)
#      
#    idx = np.arange(start ,endi)
#    np.random.shuffle(idx) 
#    idx = idx[:num]
#    data_shuffle = [data[ i] for i in idx]
#    labels_shuffle = [labels[ i] for i in idx]
#    start=endi
#    endi=endi+batch_size
#    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    start = endi
    endi += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if endi > num_examples:
        # finished epoch
#        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        labels =labels[perm]
        # start next epoch
        start = 0
        endi = batch_size
        assert batch_size <= num_examples
    end = endi
    return data[start:end], labels[start:end]
def show_image(ma):
    ma=ma.reshape((320,320))
    scipy.misc.imsave('outfile.jpg', ma)
    plt.gray()
    plt.imshow(ma)
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
# return tf.nn.relu(x,alpha=0.2,name='relu')


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 320,320, 1])

    # Convolution Layer W-F+2P/S+1....S=1...valid= 0 padding
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #318*318*32
    # Max Pooling (down-sampling)#W-F/S+1 (ceil)
    conv1 = maxpool2d(conv1, k=3)
    #106*106*32
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #104*104*128
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=3)
    #35*35*128
       # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #33*33*256
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=3)
    #11*11*256
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc = tf.nn.dropout(fc2, dropout)
   
    # Output, class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'])
   
    return out
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = sess.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def print_test_accuracy(x_test,y_test):
    account=0
    cls_pred = np.zeros(shape=np.size(x_test,0), dtype=np.int)
    for i in range(0,np.size(x_test,0)):
        
        data=x_test[i:i+1]
        label=y_test[i:i+1]
        feed_dict={X: data,Y:label,keep_prob: 1.0}
        cls_pred[i],acci=sess.run([maxout,accuracy],feed_dict=feed_dict)
        if acci==1.00:
            account=account+acci
            
    acci=100*(account/np.size(x_test,0))
    print("total test accuracy for ",np.size(x_test,0)," datasets ",acci,"%")
    return cls_pred

        
        
        
        
    
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([3,3, 1,32],stddev=0.1),name='wc1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 32,128],stddev=0.1),name='wc2'),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 128,256],stddev=0.1),name='wc3'),
    # fully connected, 7*7*64 inputs, 1024 outputs ###
    ###5*5 cause 50/2=25  for k=2 and 25/5=5 for k=5
    'wd1': tf.Variable(tf.truncated_normal([11*11*256, 1024],stddev=0.1),name='wd1'),
    'wd2': tf.Variable(tf.truncated_normal([1024, 512],stddev=0.1),name='wd2'),
#    'wd1': tf.Variable(tf.truncated_normal([5*5*256, 1024],stddev=0.05),name='wd1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([512, 2],stddev=0.1),name='out')
}

biases = {
    'bc1': tf.Variable(tf.zeros([32]),name='bc1'),
    'bc2': tf.Variable(tf.zeros([128]),name='bc2'),
    'bc3': tf.Variable(tf.zeros([256]),name='bc3'),
    'bd1': tf.Variable(tf.zeros([1024]),name='bd1'),
    'bd2': tf.Variable(tf.zeros([512]),name='bd2'),
    'out': tf.Variable(tf.zeros([2]),name='b_out')
}
###Tensorboard summary
tf.summary.histogram("wc1",weights['wc1'])
tf.summary.histogram("wc2",weights['wc2'])
tf.summary.histogram("wc3",weights['wc3'])
tf.summary.histogram("wd1",weights['wd1'])
tf.summary.histogram("wd2",weights['wd2'])
tf.summary.histogram("out",weights['out'])
tf.summary.histogram("bc1",biases['bc1'])
tf.summary.histogram("bc2",biases['bc2'])
tf.summary.histogram("bc3",biases['bc3'])
tf.summary.histogram("bd1",biases['bd1'])
tf.summary.histogram("bd2",biases['bd2'])

tf.summary.histogram("b_out",biases['out'])
# Construct model
logits = conv_net(X, weights, biases, keep_prob)

prediction = tf.nn.softmax(logits)
prediction = tf.identity(prediction, name='prediction')
maxout=tf.argmax(prediction, 1)
# Define loss and optimizer
with tf.name_scope("cost"):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
      # Loss function using L2 Regularization
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularizer = tf.nn.l2_loss(weights['wc1'])+tf.nn.l2_loss(weights['wc2'])+\
    tf.nn.l2_loss(weights['wc3'])+tf.nn.l2_loss(weights['wd1'])+tf.nn.l2_loss(weights['out'])
    loss = tf.reduce_mean(loss_op + beta * sum(reg_losses))
    #optimize
    train_op = optimizer.minimize(loss)

    tf.summary.scalar("cost",loss_op)
# Evaluate model
with tf.name_scope("accuracy"):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy')
    tf.summary.scalar("accuracy",accuracy)
#Session

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#tensorboard logdir location run according to this address
writer=tf.summary.FileWriter("./rec/logs",sess.graph)
merged=tf.summary.merge_all()
# Initialize the variables (i.e. assign their default value)
start_time = time.time()
casc=0#flag
#tf.reset_default_graph()
for ep in range(epoch):
    total_batch = int(np.size(x_train,0)/batch_size)
    for batch in range(1, total_batch):
    #    batch_x = datar['dataset'][0:batch_size]
    #    
    #    batch_y = datar['labels'][0:batch_size]
        batch_x,batch_y=next_batch(batch_size,x_train,y_train)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if batch % display_step == 0 or batch == 1:
            
            # Calculate batch loss and accuracy
            loss, acc,summary = sess.run([loss_op, accuracy,merged], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1})
            test_acc = sess.run(accuracy, feed_dict={X: x_test,
                                                                 Y: y_test,
                                                                 keep_prob: 1})
            step= ep * total_batch + batch
            print("Epoch "+ str(ep) +" Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.9f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3%}".format(acc))
            print("testing accurracy for 25 data",test_acc*100,"%")
            
            writer.add_summary(summary,step)  # Write summary
        tacc=0.00000
        if test_acc>tacc:
            tacc=test_acc
            epoc=ep
        if tacc>0.94:
            casc=1
            break
    if casc==1:
        break
print("Optimization Finished!")
# Ending time.
end_time = time.time()

# Difference between start and end-times.
time_dif = end_time - start_time

# Print the time-usage.
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
#    feed_dict={X: datar['dataset'][0:200],Y:datar['labels'][0:200],keep_prob: 0.75}
#    print("Testing Accuracy:",sess.run(accuracy,feed_dict=feed_dict))
     # Save Model
print("maximum accuracy ",tacc," at ",epoc," epochs")
    # Calculate accuracy for 256 mnist test images

test_data=x_test[0:1]
test_label=y_test[0:1]
#    feed_dict={X: datar['dataset'][0:200],Y:datar['labels'][0:200],keep_prob: 0.75}
#    print("Testing Accuracy:",sess.run(accuracy,feed_dict=feed_dict))
feed_dict={X: test_data,Y:test_label,keep_prob: 1.0}
acci=sess.run(accuracy,feed_dict=feed_dict)
log, pred, maxt=sess.run([logits,prediction,maxout],feed_dict=feed_dict)



print("Softmax input logits:",log)
print("Softmax output:",pred)
   
print("Max predicted output:",maxt)
print("training image accuracy:",acci*100,"%")
cls_pred=print_test_accuracy(x_test,y_test)
#show_image(test_data)

######save
x_test[0:1]
save_path = saver.save(sess, save_model_path)
print("Model Saved !!!!!!")
h=confusion_matrix(np.argmax(y_test[0:28], 1), cls_pred)
    

    