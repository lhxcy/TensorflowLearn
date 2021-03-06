""" Linear Regression Example """

import tflearn
# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

#LinearRegression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear,optimizer='sgd',loss='mean_square',
								metric='R2',learning_rate=0.01)

model = tflearn.DNN(regression)
model.fit(X,Y,n_epoch=1000, show_metric=True, snapshot_epoch=False)
'''
show_metric: `bool`. Display or not accuracy at every step.
snapshot_epoch: `bool`. If True, it will snapshot model at the end
                of every epoch. (Snapshot a model will evaluate this model
                on validation set, as well as create a checkpoint if
                'checkpoint_path' specified).
'''

print("\nRegression result:")
print("Y = " + str(model.get_weights(linear.W)) +
      "*X + " + str(model.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(model.predict([3.2, 3.3, 3.4]))




