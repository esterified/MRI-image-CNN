# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:23:03 2017

@author: ESTERIFIED

"""


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

pickle_file = 'edss.pickle'
save_model_path = './edss_classification'
f = open(pickle_file, 'rb')
datar = pickle.load(f)
clsLabels = np.argmax(datar['labels'], axis=1)
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
def show_image(ma):
    ma=ma.reshape((320,320))
    scipy.misc.imsave('outfile.jpg', ma)
    plt.figure()
    plt.gray()
    plt.imshow(ma)
test_data=datar['dataset'][29:30]
test_label=datar['labels'][29:30]
tf.reset_default_graph()
print("Restoring the graph!!!")
    
loaded_graph = tf.get_default_graph()
sess=tf.Session(graph=loaded_graph)
sess.run(tf.global_variables_initializer())
loader = tf.train.import_meta_graph(save_model_path + '.meta')
loader.restore(sess, save_model_path)
#graph
loaded_x = loaded_graph.get_tensor_by_name('wc1:0')
loaded_y = loaded_graph.get_tensor_by_name('wc2:0')
# Store layers weight & bias

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': 0,
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': 0,
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wc3': 0, 
    'wd1': 0,
    'wd2': 0,
    # 1024 inputs, 10 outputs (class prediction)
    'out': 0
}

biases = {
    'bc1': 0,
    'bc2': 0,
    'bc3': 0,
    'bd1': 0,
    'bd2': 0,
    'out': 0
}
#graph
weights['wc1'] = loaded_graph.get_tensor_by_name('wc1:0')
weights['wc2'] = loaded_graph.get_tensor_by_name('wc2:0')
weights['wc3'] = loaded_graph.get_tensor_by_name('wc3:0')
weights['wd1'] = loaded_graph.get_tensor_by_name('wd1:0')
weights['wd2'] = loaded_graph.get_tensor_by_name('wd2:0')
weights['out'] = loaded_graph.get_tensor_by_name('out:0')
biases['bc1'] = loaded_graph.get_tensor_by_name('bc1:0')
biases['bc3'] = loaded_graph.get_tensor_by_name('bc3:0')
biases['bc2'] = loaded_graph.get_tensor_by_name('bc2:0')
biases['bd1'] = loaded_graph.get_tensor_by_name('bd1:0')
biases['bd2'] = loaded_graph.get_tensor_by_name('bd2:0')
biases['out'] = loaded_graph.get_tensor_by_name('b_out:0')


# Network Parameters
num_input = 320*320 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
# probability of keeping a unit active. higher = less dropout
# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input],name='x')
Y = tf.placeholder(tf.float32, [None, num_classes],name='y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob') # dropout (keep probability)
def plot_conv(dummy):
    
    
    images=dummy[0,:,:,0:25]
#    images=np.reshape(image,(121,np.shape(conn)[1],np.shape(conn)[1]))
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(5, 5)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[:,:,i].reshape(np.shape(dummy)[1],np.shape(dummy)[1]), cmap='gray')

        # Show true and predicted classes.
        if 1 :
            xlabel = i
        else:
            xlabel = "True: , Pred: "

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    
    plt.show()
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x,alpha=0.2,name='leaky_relu')


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
    fc1 = tf.nn.leaky_relu(fc1)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.leaky_relu(fc2)
    # Apply Dropout
    fc = tf.nn.dropout(fc2, dropout)
   
    # Output, class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'])
   
    return out,conv1
# Construct model
def print_test_accuracy(x_test,y_test):
    account=0
    for i in range(0,np.size(x_test,0)+1):
        
        data=x_test[i:i+1]
        label=y_test[i:i+1]
        feed_dict={X: data,Y:label,keep_prob: 1.0}
        acci=sess.run(accuracy,feed_dict=feed_dict)
        if acci==1.00:
            account=account+acci
            
    acci=100*(account/np.size(x_test,0))
    print("total test accuracy for ",np.size(x_test,0)," datasets ",acci,"%")

logits,conv1 = conv_net(X, weights, biases, keep_prob)

prediction = tf.nn.softmax(logits)
prediction = tf.identity(prediction, name='prediction')
maxout=tf.argmax(prediction, 1)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy')
test_dataa=datar['dataset'][0:1]
test_labela=datar['labels'][0:1]
##Session
#sess = tf.Session()
## Initialize the variables (i.e. assign their default value)
#sess.run(tf.global_variables_initializer())
#loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
#loaded_logits = loaded_graph.get_tensor_by_name('prediction:0')
#loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
#test_predictions = sess.run( tf.nn.top_k(loaded_logits,4), \
#        feed_dict={loaded_x:  test_data, loaded_y: test_label, loaded_keep_prob: 1.0})
#show_image(test_data)
#print(test_predictions.indices)
test_data=x_test[0:1]
test_label=y_test[0:1]

feed_dict={X: test_data,Y:test_label,keep_prob: dropout}
log, pred, maxt,conn=sess.run([logits,prediction,maxout,conv1],feed_dict=feed_dict)
feed_dicta={X: test_dataa,Y:test_labela,keep_prob: dropout}
acc=sess.run(accuracy,feed_dict=feed_dicta)
print("Prediction:",pred)

print("Softmax input logits:",log)
   
print("Max predicted output:",maxt)

#print("accuracy for 1 images:",acc*100," %")
print_test_accuracy(x_test,y_test)

plot_conv(conn)

show_image(test_data)

plt.rcdefaults()
 
objects = ('EDSS < 4', 'EDSS >4')
y_pos = np.arange(len(objects))
performance = [pred[0,0],pred[0,1]]
plt.figure() 
plt.bar(y_pos, performance, align='center', alpha=0.4)
plt.xticks(y_pos, objects)
plt.ylabel('Output probabilities')
plt.title('White Matter Lesion Classification')