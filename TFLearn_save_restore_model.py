""" An example showing how to save/restore models and retrieve weights. """
# from __future__ import absolute_import, division, print_function
import tflearn
import tensorflow as tf
#Import MINIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("data/MNIST",one_hot=True)
X = mnist.train.images
Y = mnist.train.labels
testX =mnist.test.images[:10]
testY =mnist.test.labels[:10]
model_path = "model/tflearnModel/example1/"

# # MNIST Data
# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data(one_hot=True)

#Model
input_layer = tflearn.input_data(shape=[None,784],name='input')
dense1 = tflearn.fully_connected(input_layer,128,name='dense1')
dense2 = tflearn.fully_connected(dense1,256,name='dense2')
softmax = tflearn.fully_connected(dense2,10,activation='softmax')
regression = tflearn.regression(softmax,optimizer='adam',
								learning_rate=0.001,
								loss='categorical_crossentropy')

#Define classifier,with model checkpoint(autosave)
model = tflearn.DNN(regression,checkpoint_path=model_path+'model.tfl.ckpt')

# Train model, with model checkpoint every epoch and every 200 training steps
model.fit(X,Y,n_epoch=1,
		  validation_set=(testX,testY),
		  show_metric=True,
		  snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
		  snapshot_step=500, # Snapshot (save & evalaute) model every 500 steps.
		  run_id='model_and_weights')


# ---------------------
# Save and load a model
# ---------------------

#Manually save model
model.save(model_file=model_path+'model.tf1')

#Load a model
model.load(model_file=model_path+'model.tf1')

# Or Load a model from auto-generated checkpoint
# >> model.load("model.tfl.ckpt-500")

#Resume training
model.fit(X,Y,n_epoch=1,
		  validation_set=(testX, testY),
		  show_metric=True,
		  snapshot_epoch=True,
		  run_id='model_and_weights')

#Load a model
#prediction
print("load model")
model.load(model_file=model_path+'model.tf1')
pred = model.predict(testX)
correct = tf.equal(tf.argmax(pred,1),tf.argmax(testY,1))
acc = tf.reduce_mean(tf.cast(correct,tf.float32))
# print("correct:",tf.Session().run(correct))
print("Accuracy:","{:.9f}".format(tf.Session().run(acc)))




# # ------------------
# # Retrieving weights
# # ------------------
#
# #Retrive a layer weights , by layer name:
# dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
# #Get a variable's value, using model 'get_weights' method:
# print("Dense1 layer weights:")
# print(model.get_weights(dense1_vars[0]))
# # Or using generic tflearn function:
# print("Dense1 layer biases:")
# with model.session.as_default():
#     print(tflearn.variables.get_value(dense1_vars[1]))
#
#
# # It is also possible to retrieve a layer weights through its attributes `W`
# # and `b` (if available).
# # Get variable's value, using model `get_weights` method:
# print("Dense2 layer weights:")
# print(model.get_weights(dense2.W))
# # Or using generic tflearn function:
# print("Dense2 layer biases:")
# with model.session.as_default():
#     print(tflearn.variables.get_value(dense2.b))