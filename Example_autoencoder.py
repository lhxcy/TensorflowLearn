"""
Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import MINIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("data/MNIST",one_hot=True)

#Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256
display_step = 1000
examples_to_show = 10

#Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_input = 784  # MNIST data input (img shape: 28*28)

#tf Graph input (only picure)
X = tf.placeholder("float",[None,num_input])

#Define weights
weights = {
	"encoder_h1":tf.Variable(tf.random_normal([num_input,num_hidden_1])),
	"encoder_h2":tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
	"decoder_h1":tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1])),
	"decoder_h2":tf.Variable(tf.random_normal([num_hidden_1,num_input]))
}

biases = {
	"encoder_b1":tf.Variable(tf.random_normal([num_hidden_1])),
	"encoder_b2":tf.Variable(tf.random_normal([num_hidden_2])),
	"decoder_b1":tf.Variable(tf.random_normal([num_hidden_1])),
	"decoder_b2":tf.Variable(tf.random_normal([num_input]))
}

#Building the encoder
def encoder(x):
	#Encoder Hidden layer with sigmoid activation
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights["encoder_h1"]),biases["encoder_b1"]))
	#endoder hidden layer withsigmoid activaion
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights["encoder_h2"]),biases["encoder_b2"]))
	return layer_2

#Building the decoder
def decoder(x):
	#Decoder Hidden layer with sigmoid activation
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights["decoder_h1"]),biases["decoder_b1"]))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights["decoder_h2"]),biases["decoder_b2"]))
	return layer_2

#Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

#Prediction
y_pred = decoder_op
#Targets are input data
y_true = X

#Define loss and optimizer,minimize the squared error
loss_op = tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_op)

#Initialize the variables
init = tf.global_variables_initializer()

#Start Training
with tf.Session() as sess:
	sess.run(init)
	for i in range(1,num_steps+1):
		batch_x, _ =mnist.train.next_batch(batch_size)
		_, loss = sess.run([optimizer,loss_op],feed_dict={X:batch_x})
		if i % display_step == 0 or i == 1:
			print("step %i: Minibatch Loss: %f" %(i,loss))

	print("Optimization finished")

	#Testing
	n = 4
	canvas_orig = np.empty((28*n,28*n))
	# print("canvas_orig:\n",canvas_orig)
	canvas_recon = np.empty((28*n,28*n))
	for i in range(n):
		batch_x,_ = mnist.test.next_batch(n)
		g = sess.run(decoder_op,feed_dict={X:batch_x})

		#Display original images
		for j in range(n):
			#Draw the original digits
			canvas_orig[i*28:(i+1)*28,j*28:(j+1)*28] = \
				batch_x[j].reshape([28,28])

		for j in range(n):
			# Draw the original digits
			canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
				g[j].reshape([28, 28])

	print("Original Images")
	plt.figure(figsize=(n,n))
	'''
	figsize : tuple of integers, optional, default: None
        width, height in inches. If not provided, defaults to rc
        figure.figsize.
	'''
	plt.imshow(canvas_orig,origin="upper",cmap="gray")
	# plt.show()

	print("Reconstructed Images")
	plt.figure(figsize=(n,n))
	plt.imshow(canvas_recon,origin="upper",cmap="gray")
	plt.show()

'''
gray 返回线性灰度色图
imshow(X, cmap=None, norm=None, aspect=None,
interpolation=None, alpha=None, vmin=None,
vmax=None, origin=None, extent=None, shape=None,
filternorm=1, filterrad=4.0, imlim=None,
resample=None, url=None, hold=None, data=None, **kwargs)
其中，X变量存储图像，可以是浮点型数组、unit8数组以及PIL图像，如果其为数组，则需满足一下形状：
    (1) M*N      此时数组必须为浮点型，其中值为该坐标的灰度；
    (2) M*N*3  RGB（浮点型或者unit8类型）
    (3) M*N*4  RGBA（浮点型或者unit8类型）
'''












