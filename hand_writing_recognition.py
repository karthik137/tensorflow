
'''
classification using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:

    It should succeed in less than 10 epochs, so it is okay to change epochs to 10, but nothing larger
    When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
    If you add any additional variables, make sure you use the same names as the ones used in the class

'''

import tensorflow as tf 


### Callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True


callbacks = myCallback()


mnist = tf.keras.datasets.mnist

print(mnist)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train[0])

#print(y_train)

'''
Normalize the data sets
'''

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(2048, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


model.fit(x_train, y_train, epochs=8, callbacks=[callbacks])

classifications = model.predict(x_test)

print(classifications)

