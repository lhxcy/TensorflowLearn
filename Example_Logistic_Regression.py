import tensorflow as tf
#Import MINIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("data/MNIST",one_hot=True)

#Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

#tf Graph Input
x = tf.placeholder(tf.float32,[None,784])#mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32,[None,10])#0-9 digist recognition => 10 classes

#Set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Construct model
# pred = tf.nn.softmax(tf.matmul(x,W)+b)#softmax
# pred = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(x,W)+b))#softmax
pred = tf.nn.sigmoid(tf.matmul(x,W)+b)#sigmoid

#Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)+(1-y)*tf.log(1-pred),reduction_indices=1))
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
# 有一个reduction_indices参数，表示函数的处理维度
#Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Initialize the variables(i.e. assign theri default value)
init = tf.global_variables_initializer()

#Start training
with tf.Session() as sess:
	sess.run(init)
	#Training Cycle
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		#Loop over all batches
		for i in range(total_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)
			#Fit training using batch data
			_,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
			#Compute average loss
			avg_cost += c/total_batch
		#Display logs per epoch step
		if (epoch+1) % display_step == 0:
			print("Epoch:","%04d" % (epoch+1),"cost=","{:.9f}".format(avg_cost))
	print("Optimization Finished")

	#Test model
	correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	#Caculate accuracy for 3000 examples
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	# tf.cast(x, dtype, name=None)将x的数据格式转化成dtype.
	# print("Accruacy:",accuracy.eval({x:mnist.test.images[:3000],y:mnist.test.labels[:3000]}))
	print("Accruacy:",sess.run(accuracy,feed_dict={x:mnist.test.images[:3000],y:mnist.test.labels[:3000]}))
# 	如果你有一个Tensor t，在使用t.eval()时，等价于：tf.get_default_session().run(t).
# 	run可以同时执行多个Tensor，eval每次只能一个



