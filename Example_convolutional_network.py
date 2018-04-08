import tensorflow as tf
#Import MINIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("data/MNIST",one_hot=True)

#Training Parameters
learning_rate = 0.001
num_steps = 2000                 #////////////代码跑不通
batch_size = 128

#Network Parameters
num_input = 784
num_classes = 10
dropout = 0.25

#Creat the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
	#define a scope for reusing the variables
	with tf.variable_scope("ConvNet",reuse=reuse):
		#TF Estimator input is a dict,in case of multiples inputs
		x = x_dict["images"]

		#MNIST datainput is a 1-D vector of 784 features
		#Reshape to match picture format [Height*Width*Channel]
		#Tensor input become 4-D:[bach size,Height,Width,Channel]
		x = tf.reshape(x,shape=[-1,28,28,1])
		#Convolution Layer with 32 filters and a kernel size of 5
		conv1 = tf.layers.conv2d(x,32,5,activation=tf.nn.relu)
		# conv1 = tf.layers.conv2d(x,32,[5,4,3],activation=tf.nn.relu)
		#Max Pooling (Down-sampling)with strides of 2 and kernel size of 2
		conv1 = tf.layers.max_pooling2d(conv1,2,2)

		#Convolution Layer with 64 filters and a kernel size of 3
		conv2 = tf.layers.conv2d(conv1,64,3,activation=tf.nn.relu)
		conv2 = tf.layers.max_pooling2d(conv2,2,2)
		#Flatten the data to a 1-D vector for the fully connected layer
		fc1 = tf.contrib.layers.flatten(conv2)
		#Fully connected layer (in tf contrib folder for now)
		fc1 = tf.layers.dense(fc1,1024)
		#Apply Dropout(if istraining is False,dropout is not applied)
		fc1 = tf.layers.dropout(fc1,rate=dropout,training=is_training)

		#Output layer,class prediction
		out = tf.layers.dense(fc1,n_classes)
		return out

#Define the model fuction(following TF Estimator Template)
def model_fn(features,labels,mode):
	#Build the neural network
	#Because Dropout have different behavior a training and prediction
	#time,we need to creat 2 distinct computation graphs that still share
	#the same weights.
	logits_train = conv_net(features,num_classes,dropout,reuse=False,
							is_training=True)
	logits_test = conv_net(features,num_classes,dropout,reuse=True,
						   is_training=False)
	#predictions
	pred_classes = tf.argmax(logits_test,axis=1)
	pred_probas = tf.argmax(logits_train,axis=1)

	#If prediction mode, early return
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode,prediction=pred_classes)

	#Define loss and optimizer
	# loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train,labels=tf.cast(labels,dtype=tf.int32)))
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pred_probas,logits=tf.cast(labels,dtype=tf.int32)))
	# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.int32), logits=pred_probas))
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
	# tf.train.get_global_step() 学习速率变化时会用到
	# learning_rate = tf.train.exponential_decay(0.1, global_steps, 10, 2, staircase=False)

	#evalue the accuracy of the model
	acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_classes)

	#TF Estimators requires to return a EstimatorSpec,that specify
	#the different ops for training ,evaluating,...
	estim_specs = tf.esimator.EstimatorSpec(
		mode=mode,
		predictioin=pred_classes,
		train_op=train_op,
		eval_metric_ops={"accurate":acc_op}
	)
	return estim_specs

#Build the Estimator
model = tf.estimator.Estimator(model_fn)
#Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"images":mnist.train.images},y=mnist.train.labels,
	batch_size=batch_size,num_epochs=None,shuffle=True)

#train the model
model.train(input_fn,steps=num_steps)

#Evaluate the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
	x = {"images":mnist.test.images},y=mnist.test.labels,
	batch_size=batch_size,shuffle=False)

#Use the Estiator 'evaluate'method
e = model.evaluate(input_fn)

print("Testing Accuracy:",e["accuracy"])


