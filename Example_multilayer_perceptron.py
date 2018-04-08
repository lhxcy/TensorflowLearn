import tensorflow as tf
#Import MINIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("data/MNIST",one_hot=True)

#Parameters
learning_rate= 0.01
training_epochs = 15
batch_size = 100
display_step = 1

#Network Parameters
n_hadden_1 = 256 #1st layer number of neurons
n_hadden_2 = 256 #2nd layer number of neurons
n_input = 784 #MNIST daa input
n_classes = 10 #MNIST total classes

# tf Graph input
X = tf.placeholder("float",[None,n_input])
Y = tf.placeholder("float",[None,n_classes])

# Store layers weight & bias
weights = {
	"h1": tf.Variable(tf.random_normal([n_input,n_hadden_1])),
	"h2": tf.Variable(tf.random_normal([n_hadden_1,n_hadden_2])),
	"out": tf.Variable(tf.random_normal([n_hadden_2,n_classes]))
}
bias = {
	"b1": tf.Variable(tf.random_normal([n_hadden_1])),
	"b2": tf.Variable(tf.random_normal([n_hadden_2])),
	"out": tf.Variable(tf.random_normal([n_classes]))
}

#Creat model
def multilayer_perceptron(x):
	#Hidden fully connected layer with 256 neurons
	# layer_1 = tf.add(tf.matmul(x,weights["h1"]),bias["b1"])
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights["h1"]),bias["b1"]))
	#Hidden fully connected layer with 256 neurons
	# layer_2 = tf.add(tf.matmul(layer_1,weights["h2"]),bias["b2"])
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights["h2"]),bias["b2"]))
	#Output fully connected layer with a neuron for each class
	# out_layer = tf.matmul(layer_2,weights["out"])+bias["out"]
	out_layer = tf.nn.sigmoid(tf.matmul(layer_2,weights["out"])+bias["out"])
	return out_layer

#construct model
logits = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
# tf.nn.softmax_cross_entropy_with_logits具体的执行流程大概分为两步：
# 1:第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，
# 输出就是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）
# 2:第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss=loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

#Start train
with tf.Session() as sess:
	sess.run(init)
	#Training Cycle
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_x,batch_y = mnist.train.next_batch(batch_size)
			_,c = sess.run([train_op,loss_op], feed_dict={X:batch_x,Y:batch_y})
			avg_cost += c/total_batch
		if (epoch+1) % display_step == 0:
			print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

	print("Optimization Finished")
	#test model
	pred = tf.nn.softmax(logits)#Apply softmax to logits
	correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	print("Accuracy:",accuracy.eval(feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
	# print("Accruacy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

