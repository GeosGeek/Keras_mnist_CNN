# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:54:11 2019

@author: Matt
"""

import keras
from keras.datasets import mnist

# Split the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
# Plotting the first image in the dataset
plt.imshow(x_train[0])

# Checking the dimensions of the first image in the dataset
x_train[0].shape

# Reshape the data
# We want 60,000 images in train set, 10,000 in test set. 28x28 are the dimensions, 1 represents grayscale
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# One-hot encode the target column, binary output for each input
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Printing the first element at index 6, denoting a digit 5
y_train[0]

# Building the model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

# Compile the model using accuracy to measure performance
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=3)






