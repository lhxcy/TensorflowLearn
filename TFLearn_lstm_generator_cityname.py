"""
城市名称生成，使用 LSTM 网络生成新的美国城市名
"""
import os
from six import moves
import ssl
import tflearn
from tflearn.data_utils import *
path = 'data/US_Cities.txt'
url = "https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt"
if not os.path.isfile(path):
	print("download US_Cities.txt")
	moves.urllib.request.urlretrieve(url, path)

maxlen = 20

string_utf8 = open(path,'r',encoding='utf-8').read()
# file.read()  #读全部
# print(type(string_utf8))
# print(len(string_utf8))
# print(string_utf8)
X,Y,char_idx = string_to_semi_redundant_sequences(string_utf8,seq_maxlen=maxlen,redun_step=3)
# print(X[0])
# print(Y[0])
""" string_to_semi_redundant_sequences.

    Vectorize a string and returns parsed sequences and targets, along with
    the associated dictionary.

    Arguments:
        string: `str`. Lower-case text from input text file.
        seq_maxlen: `int`. Maximum length of a sequence. Default: 25.
        redun_step: `int`. Redundancy step. Default: 3.
        char_idx: 'dict'. A dictionary to convert chars to positions. Will be automatically generated if None

    Returns:
        A tuple: (inputs, targets, dictionary)
    """

#Network
net = tflearn.input_data(shape=[None,maxlen,len(char_idx)])
net = tflearn.lstm(net,512,return_seq=True)
net = tflearn.dropout(net,0.5)
net = tflearn.lstm(net,512)
net = tflearn.dropout(net,0.5)
net = tflearn.fully_connected(net,len(char_idx),activation='softmax')
net = tflearn.regression(net,optimizer='adam',loss='categorical_crossentropy',
						 learning_rate=0.001)
model = tflearn.SequenceGenerator(net,dictionary=char_idx,
								  seq_maxlen=maxlen,
								  clip_gradients=5.0,
								  checkpoint_path='model/tflearnModel/example2/model_us_cities')

for i in range(40):
	seed = random_sequence_from_string(string_utf8,maxlen)
	model.fit(X,Y,validation_set=0.1,batch_size=128,
			  n_epoch=1,run_id='us_cities')
	print("--Testing--")
	print("-- Test with temperature of 1.2 --")
	print(model.generate(30,temperature=1.2,seq_seed=seed).encode('utf-8'))
	""" Generate.

	        Generate a sequence. Temperature is controlling the novelty of
	        the created sequence, a temperature near 0 will looks like samples
	        used for training, while the higher the temperature, the more novelty.
	        For optimal results, it is suggested to set sequence seed as some
	        random sequence samples from training dataset.

	        Arguments:
	            seq_length: `int`. The generated sequence length.
	            temperature: `float`. Novelty rate.
	            seq_seed: `sequence`. A sequence used as a seed to generate a
	                new sequence. Suggested to be a sequence from data used for
	                training.
	            display: `bool`. If True, print sequence as it is generated.

	        Returns:
	            The generated sequence.

	        """
	print("-- Test with temperature of 1.0 --")
	print(model.generate(30, temperature=1.0, seq_seed=seed).encode('utf-8'))
	print("-- Test with temperature of 0.5 --")
	print(model.generate(30, temperature=0.5, seq_seed=seed).encode('utf-8'))



