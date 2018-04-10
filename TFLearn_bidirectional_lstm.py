"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
"""
import tflearn
from tflearn.data_utils import to_categorical,pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn,BasicLSTMCell
from tflearn.layers.estimator import regression

#IMDB Dataset loading
train, valid, test = imdb.load_data(path='data/imdb.pk1', n_words=20000,
							   valid_portion=0.1)

trainX,trainY = train
validX, validY = valid

#Data preprocessing
#Sequence padding
trainX = pad_sequences(trainX,maxlen=200,value=0.)
# Returns:x: `numpy array` with dimensions (number_of_sequences, maxlen)

validX = pad_sequences(validX,maxlen=200,value=0.)

#Converting labels to binary vectors
trainY = to_categorical(trainY)
"""
def to_categorical(y, nb_classes=None)

Convert class vector (integers from 0 to nb_classes)
to binary class matrix, for use with categorical_crossentropy.

Arguments:
	y: `array`. Class vector to convert.
	nb_classes: `int`. The total number of classes.
"""
validY = to_categorical(validY)

#Network building
net = input_data(shape=[None,200])
net = embedding(net,input_dim=20000,output_dim=128)
net = bidirectional_rnn(net,BasicLSTMCell(128),BasicLSTMCell(128))
net = dropout(net,0.5)
net = fully_connected(net,2,activation='softmax')
net = regression(net,optimizer='adam',loss='categorical_crossentropy')

#Traing
model = tflearn.DNN(net,clip_gradients=0.,tensorboard_verbose=2)
model.fit(trainX,trainY,validation_set=0.1,show_metric=True,batch_size=64)



