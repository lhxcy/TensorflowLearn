import numpy as np
import tensorflow as tf

#Input MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST",one_hot=True)

#In this example, we limit mnist data
Xtr,Ytr = mnist.train.next_batch(5000)#5000 for training (nn candidates)
Xte,Yte = mnist.test.next_batch(200)#200 for testing

#tf Graph Input
xtr = tf.placeholder("float",[None,784])
xte = tf.placeholder("float",[784])

#Nearest Neighbor calculation using L1 Distance
#Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),reduction_indices=1)
#Prediction: Get min distance index (Nearest neighbor)
# df.mean(axis=1),我们将得到按行计算的均值
# df.mean(axis=0),我们将得到按列计算的均值
pred = tf.argmin(distance,axis=0)

accuracy = 0

#Initialize the variables(i.e. assign their default value)
init = tf.global_variables_initializer()

#Start training
with tf.Session() as sess:
	sess.run(init)

	#loop over test data
	for i in range(len(Xte)):
		#Get neares neighbor
		nn_index = sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
		#Get nearest neighbor class label and compare it to its true label
		print("Test",i,"Prediction: ",np.argmax(Ytr[nn_index]),"True Class: ",np.argmax(Yte[i]))
		#Test 189 Prediction:  [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.] True Class:  [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
		#Calculae accuracy
		if np.argmax(Ytr[nn_index]==np.argmax(Yte[i])):
			accuracy += 1./len(Xte)
	print("Done")
	print("Accuracy: ",accuracy)