"""
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import MINIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("data/MNIST",one_hot=True)

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "model/tensorflowModel/example1/model.ckpt"

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#Creat model
def mutilayer_perception(x,weights,biases):
	#Hidden layer with RELU activation
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['h1']),biases['b1']))
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
	out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
	return out_layer

#Construct model
logits = mutilayer_perception(X,weights,biases)
pred = tf.nn.softmax(logits)

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Initialize the variables
init = tf.global_variables_initializer()

#'Saver' op to save and restore all the variables
saver = tf.train.Saver()


#Running first session
print("Starting 1st session")
with tf.Session() as sess:
	sess.run(init)
	#Training cycle
	for epoch in range(3):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		#Loop over all batches
		for i in range(total_batch):
			batch_x,batch_y = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer,cost],feed_dict={X:batch_x,Y:batch_y})
			avg_cost += c/total_batch
		#display logs per eopch step
		if epoch % display_step == 0:
			print("Epoch:","%04d"% (epoch+1),"cost=","{:.9f}".format(avg_cost))

	print("First Optimization Finished!")

	#Test model
	# correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	print("Accuracy:",accuracy.eval(feed_dict={X:mnist.test.images,Y:mnist.test.labels}))

	#Save model weights to disk
	save_path = saver.save(sess,model_path)
	print("Model saved in file: %s" % save_path)

#Running a new session
print("Starting 2nd session...")
with tf.Session() as sess:
	sess.run(init)
	#Restore model weights from previously saved model
	saver.restore(sess,model_path)
	print("Model restored from file: %s" % save_path)

	#Resume training
	for epoch in range(7):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_x,batch_y = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer,cost],feed_dict={X:batch_x,Y:batch_y})
			avg_cost += c/total_batch
		if epoch % display_step == 0:
			print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

	print("Second Optimization Finished!")

	#Test model
	# correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))



