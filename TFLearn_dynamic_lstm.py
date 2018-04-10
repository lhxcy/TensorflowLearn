"""
Simple example using a Dynamic RNN (LSTM) to classify IMDB sentiment dataset.
Dynamic computation are performed over sequences with variable length.
"""
import tflearn
from tflearn.data_utils import to_categorical,pad_sequences
from tflearn.datasets import imdb

#IMDB Dataset loading
train, valid, test = imdb.load_data(path='data/imdb.pk1', n_words=10000,
							   valid_portion=0.1)

trainX,trainY = train
validX, validY = valid

#Data preprocessing
#Sequence padding
trainX = pad_sequences(trainX,maxlen=100,value=0.)
# Returns:x: `numpy array` with dimensions (number_of_sequences, maxlen)

validX = pad_sequences(validX,maxlen=100,value=0.)

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
net = tflearn.input_data(shape=[None,100])
#Masking is not required for embedding, sequence length is computed prior
#to the embedding op and assigned as 'seq_length' attribute to the return Tensor
net = tflearn.embedding(net,input_dim=10000,output_dim=128)
net = tflearn.lstm(net,128,dropout=0.8,dynamic=True)
net = tflearn.fully_connected(net,2,activation='softmax')
net = tflearn.regression(net,optimizer='adam',learning_rate=0.001,
						 loss='categorical_crossentropy')

#Training
model = tflearn.DNN(net,tensorboard_verbose=0)
model.fit(trainX,trainY,validation_set=(validX,validY),show_metric=True,
		  batch_size=32)







