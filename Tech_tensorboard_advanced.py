"""
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
"""
import tensorflow as tf

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST",one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = 'logs/MNIST/exampleAdv/'

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# tf Graph Input
# mnist data image of shape 28*28=784
X = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
Y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')
}

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram("relu1", layer_1)#直方图
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("relu2", layer_2)
    # Output layer
    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    return out_layer



#Encapsulating all ops into scopes,making Tensorboard's Graph
#visualization more convenient
with tf.name_scope("Model"):
	pred = multilayer_perceptron(X,weights,biases)

with tf.name_scope("Loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))

with tf.name_scope("SGD"):
	optimizer =tf.train.GradientDescentOptimizer(learning_rate)
	#op to calculate every variable gradient
	grads = tf.gradients(loss,tf.trainable_variables())
	grads = list(zip(grads,tf.trainable_variables()))
	#tf.trainable_variables 返回所有 当前计算图中 在获取变量时未标记 trainable=False 的变量集合
	#tf.trainable_variables返回的是需要训练的变量列表
	# tf.all_variables返回的是所有变量的列表

	#Op to update all variables according to their gradient
	apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope("Accuracy"):
	acc = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	acc = tf.reduce_mean(tf.cast(acc,tf.float32))

#Initialize the variables
init = tf.global_variables_initializer()

#Create a summary to monitor cost tensor
tf.summary.scalar("loss",loss)

tf.summary.scalar("accuracy",acc)

#Summary all gradients
for grad,var in grads:
	tf.summary.histogram(var.name,var)

#Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

#Start training
with tf.Session() as sess:
	sess.run(init)
	summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
	#Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples / batch_size)
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# Run optimization op (backprop), cost op (to get loss value)
			# and summary nodes
			_, c, summary = sess.run([apply_grads, loss, merged_summary_op],
									 feed_dict={X: batch_xs, Y: batch_ys})
			# Write logs at every iteration
			summary_writer.add_summary(summary, epoch * total_batch + i)
			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if (epoch + 1) % display_step == 0:
			print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

	print("Optimization Finished!")

	# Test model
	# Calculate accuracy
	print("Accuracy:", acc.eval({X: mnist.test.images, Y: mnist.test.labels}))

	print("Run the command line:\n" \
		  "--> tensorboard --logdir=logs/MNIST/exampleAdv/" \
		  "\nThen open http://0.0.0.0:6006/ into your web browser")







