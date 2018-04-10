from __future__ import division, print_function, absolute_import
"""
The dataset is stored in a CSV file, so we can use the TFLearn load_csv() function to
 load the data from the CSV file into a python list.
 We specify the 'target_column' argument to indicate that our labels (survived or not)
 are located in the first column (id: 0). The function will return a tuple: (data, labels).
"""
import numpy as np
import tflearn

#DownLoad the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

#loadCSVfile,indicate that the first column represent labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv',target_column=0,
						categorical_labels=True,n_classes=2)

'''
Preprocessing Data

Data are given 'as is' and need some preprocessing to be ready for use in our deep neural network classifier.
First, we will discard the fields that are not likely to help in our analysis.
For example, we make the assumption that the 'name' field will not be very useful in our task,
since a passenger's name and his or her chance of surviving are probably not correlated.
With such thinking, we can go ahead and discard the 'name' and 'ticket' fields.
Then, we need to convert all our data to numerical values,
because a neural network model can only perform operations over numbers.
However, our dataset contains some non-numerical values, such as 'name' and 'sex'. Because 'name' is discarded,
we just need to handle the 'sex' field. In this simple case, we will just assign '0' to males and '1' to females.

example:
survived	pclass	name							sex		age		sibsp	parch	ticket		fare
1			1		Aubart, Mme. Leontine Pauline	female	24		0		0		PC 17477	69.3000
'''
# Here is the preprocessing function:
#Preprocessing function
def preprocess(passengers,columns_to_delete):
	#Sort by descending is and delete column
	for column_to_delete in sorted(columns_to_delete,reverse = True):
		[passenger.pop(column_to_delete) for passenger in passengers]
	# print(type(passengers[0]))
	for i in range(len(passengers)):
		# Converting 'sex' field to float (id is 1 after removing labels column)
		passengers[i][1] = 1. if passengers[i][1] == 'female' else 0.
	print(np.array(passengers,dtype=np.float32))
	return np.array(passengers,dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore = [1,6]
#Preprocess data
data = preprocess(data,to_ignore)

'''
Build a Deep Neural Network

We are building a 3-layer neural network using TFLearn. First, we need to specify the shape of our input data.
In our case, each sample has a total of 6 features, and we will process samples per batch to save memory.
So our data input shape is [None, 6] ('None' stands for an unknown dimension, so we can change the total
number of samples that are processed in a batch).
'''
# Build neural network
net = tflearn.input_data(shape=[None,6])
net = tflearn.fully_connected(net,32)
net = tflearn.fully_connected(net,32)
net = tflearn.fully_connected(net,2,activation='softmax')
net =tflearn.regression(net)

'''
Training

TFLearn provides a model wrapper ('DNN') that automatically performs neural network classifier tasks,
such as training, prediction, save/restore, and more. We will run it for 10 epochs
(i.e., the network will see all data 10 times) with a batch size of 16.
'''

#Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

'''
Try the Model
It's time to try out our model.
For fun, let's take Titanic movie protagonists
(DiCaprio and Winslet) and calculate their chance of surviving (class 1).
'''

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])





