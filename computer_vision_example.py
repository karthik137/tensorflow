import tensorflow as tf 
import matplotlib.pyplot as plt

# Print tensor flow version
print(tf.__version__)

# Import fashion MNIST data (NOTE: MNIST Dataset is directly available in tensorflow)

mnist = tf.keras.datasets.fashion_mnist

# Print the dataset
print(mnist)

# Load training set and testing set

#(training_images, training_labels)

#training_set, testing_set = mnist.load_data()

#print(training_set[1])

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

### Lets take a look at different indices in the array

#plt.imshow(training_images[0])
#plt.show()
print(training_labels[0])
print(training_images[0])


# Normalize the images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Design the model


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(1024, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

'''
Sequential: That defines a SEQUENCE of layers in the neural network

Flatten: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.

Dense: Adds a layer of neurons

Each layer of neurons need an activation function to tell them what to do. There's lots of options, but just use these for now.

Relu effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

Softmax takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

'''

model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

#model.evaluate(test_images, test_labels)

### Exploration exercises

### Check classification

classifications = model.predict(test_images)

print(classifications[0])

print(test_labels)
