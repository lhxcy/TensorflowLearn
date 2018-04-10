"""
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
"""
import tensorflow as tf

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST",one_hot=True)

#Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_epoch = 1
logs_path = 'logs/MNIST/example/'

#tf Graph Input
#mnist data image of shape 28*28=784
X = tf.placeholder(tf.float32,[None,784],name="InputData")
Y = tf.placeholder(tf.float32,[None,10],name="LabrlData")

#Set model weights
W = tf.Variable(tf.zeros([784,10]),name="Weights")
b = tf.Variable(tf.zeros([10]),name="Bias")

#Construct model and encapsulating all ops into scopes,making
#Tensorboard's Graph visualization more convenient
with tf.name_scope("Model"):
	#Model
	pred = tf.nn.softmax(tf.matmul(X,W)+b)
with tf.name_scope("Loss"):
	#Minimize error using cross entropy
	cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))
with tf.name_scope("SGD"):
	#Gradient Descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope("Accuracy"):
	#Accuracy
	acc = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	acc = tf.reduce_mean(tf.cast(acc,tf.float32))
#Initialize the variables
init = tf.global_variables_initializer()

#Create a summary to monitor cost tensor
tf.summary.scalar("loss",cost)
#Create a summary to monior accuracy tensor
tf.summary.scalar("accuracy",acc)
#Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

#Start training
with tf.Session() as sess:
	sess.run(init)

	#Op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
	'''ummary_waiter = tf.summary.FileWriter("log",tf.get_default_graph())
	log是事件文件所在的目录，这里是工程目录下的log目录。第二个参数是事件文件要记录的图，也就是tensorflow默认的图。
	'''
	#Training Cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_x,batch_y = mnist.train.next_batch(batch_size)
			#Run optimization op(backprop),cost op(to get loss value)
			#and summary nodes
			_, c, summary = sess.run([optimizer,cost,merged_summary_op],feed_dict={X:batch_x,Y:batch_y})
			#Write logs at every iteration
			summary_writer.add_summary(summary,epoch*total_batch+i)
			#Compute average loss
			avg_cost += c/total_batch
		if (epoch+1) % display_epoch == 0:
			print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
	print("Optimization Finished")

	#Test Model
	#Caculate accuracy
	print("Accuracy:",acc.eval(feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
	print("Run the command line:\n"
		  "-->tensorboard --logdir=logs/MNIST/example/"
		  "\nThen open http://0.0.0.0:6006/ into your web browser")