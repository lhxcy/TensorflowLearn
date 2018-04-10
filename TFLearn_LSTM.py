"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
循环神经网络（LSTM），应用 LSTM 到 IMDB 情感数据集分类任
"""
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
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
net = tflearn.input_data([None,100])
net = tflearn.embedding(net,input_dim=10000,output_dim=128)
'''
tflearn.embedding(net,input_dim=10000,output_dim=128)
 Input:
        2-D Tensor [samples, ids].

    Output:
        3-D Tensor [samples, embedded_ids, features].

    Arguments:
        incoming: Incoming 2-D Tensor.
        input_dim: list of `int`. Vocabulary size (number of ids).
        output_dim: list of `int`. Embedding size.
'''
net = tflearn.lstm(net,128,dropout=0.8)
'''
tflearn.lstm(net,128,dropout=0.8)
Input:
        3-D Tensor [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].
'''
net = tflearn.fully_connected(net,2,activation='softmax')
net =tflearn.regression(net,optimizer='adam',learning_rate=0.001,
						loss='categorical_crossentropy')

#Training
model = tflearn.DNN(net,tensorboard_verbose=0)
model.fit(trainX,trainY,validation_set=(validX,validY),show_metric=True,
		  batch_size=32)











