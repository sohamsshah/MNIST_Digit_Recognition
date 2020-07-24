# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:51:20 2019

@author: Soham Shah
"""

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train =tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

plt.imshow(x_train[0])

plt.show()
print(x_train[0])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


model.compile(optimizer= 'sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)


model.save('hello world')

new_model = tf.keras.models.load_model('hello world')

predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])

plt.imshow(x_test[1])
print(np.argmax(predictions[1]))









