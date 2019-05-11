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

plt.imshow(training_images[0])
plt.show()
print(training_labels[0])
print(training_images[0])


# Normalize the images
training_images = training_images / 255.0
test_images = test_images / 255.0

