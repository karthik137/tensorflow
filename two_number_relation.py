
'''
Hello world program of tensor flow

float hw_function(float x){
    float y = (2 * x) - 1;
    return y;
}

So how would you train a neural network to do the equivalent task? Using data! By feeding it with a set of Xs, and a set of Ys, it should be able to figure out the relationship between them.

'''

import tensorflow as tf
import numpy as np
from tensorflow import keras


'''

Define and Compile the Neural Network

Next we will create the simplest possible neural network. It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.

'''

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

'''
we have to specify 2 functions, a loss and an optimizer.
'''
model.compile(optimizer='sgd', loss='mean_squared_error')

'''

Providing the Data

Next up we'll feed in some data. In this case we are taking 6 xs and 6ys. You can see that the relationship between these is that y=2x-1, so where x = -1, y=-3 etc. etc.

A python library called 'Numpy' provides lots of array type data structures that are a defacto standard way of doing it. We declare that we want to use these by specifying the values as an np.array[]

'''

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

'''
Training the Neural Network
'''

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))