import tensorflow as tensor 
import numpy as np 
from tensorflow import keras 

'''
Housing Price Condition

One house cost --> 50K + 50K per bedroom

2 bedroom cost --> 50K + 50 * 2 = 150K
3 bedroom cost --> 50K + 50 * 3 = 200K
.
.
.
7 bedroom cost --> 50K + 50 * 4 = 400K

'''


'''
Training set to be given
xs = [100,150,200,250,300,350]
ys = [1,2,3,4,5,6]

'''



# Create model
model = tensor.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0],dtype=float)
ys = np.array([100.0,150.0,200.0,250.0,300.0,350.0],dtype=float)

# Train the neural network
model.fit(xs, ys, epochs=1000)

print("Predicting for house with number of bedrooms = 7")
print(model.predict([7.0]))
