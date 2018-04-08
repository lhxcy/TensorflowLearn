import tensorflow as tf
#basic constant operations
#The value returned by the consructor represents the optput of the constant op
a = tf.constant(2)
b = tf.constant(3)

#launch the default graph
with tf.Session() as sess:
	print("a: %i" %sess.run(a),"b: %i" %sess.run(b))
	print("Addition with constants: %i" %sess.run(a+b))
	print("Multiplication with constants: %i" %sess.run(a*b))





#Basic operations with variable as graph input
#The value returned by the constructor represents the output of the Variable op(define as input when running session)
#tf Graph input

c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)

#Define some operations
add = tf.add(a,b)
mul = tf.multiply(a,b)

#launch the default graph
with tf.Session() as sess:
	#run every operation with variable input
	print("Addition with variables: %i" %sess.run(add,feed_dict={a:2,b:3}))
	print("Multiplication with variables: %i" % sess.run(mul,feed_dict={a:2,b:3}))




#----------------------
#More in details
#Matrix Multiplication from TensorFlow official tutorial
#
#Creat a Constant op that produces a 1X2 matrix. The op is
#added as a node to the default graph
#
#The value returned by the constructor represents the output
#of theconstant op
matrix1 = tf.constant([[3.,3.]])

#Creat another constant that produces a 2X1 matrix
matrix2 = tf.constant([[2.],[2.]])

#creat a Matmul op that takes 'matrix1' and 'matrix2' as input
#The returned value, 'product', represents the result of the atrix multiplication
product = tf.matmul(matrix1,matrix2)

#To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.This indicates to the call
# that we want to get the output of the matmul op back.

# All inputs needed by the op are run automatically by the session. They
# typically are run in parallel.

# the call 'run(prodect)' thus causes the execution of threes ops in the
# graph: the two constants and matmul

# The output of the op is returned in 'result' as a numpy 'ndarray' object
with tf.Session() as sess:
	result = sess.run(product)
	print(result)