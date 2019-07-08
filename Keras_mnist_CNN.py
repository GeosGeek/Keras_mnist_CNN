import keras
from keras.datasets import mnist

# Split the mnists dataset to train/test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt

# Plot the first image in the dataset
plt.imshow(x_train[0])

# Check the dimensions of the first image
# All images in this dataset are 28x28 pixels, this does not happen with real-world datasets. Each image must be reshaped. 
x_train[0].shape

# Reshape the data to fit the model.
# We want 60,000 images in the train set and 10,000 in the test set. 28x28 are the dimensions of the image,
# 1 represents the images as greyscale. 
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# one-hot encode the target column
# A column will be created for each output category and binary variable is inputted for each category. 
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]

# Build the model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile the model using accuracy to measure performance
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Training
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

# Using the model to predict.
# The predict funtion returns an array of 10 numbers that are probabilities of the unput being each digit. 
# The array index with the highest number is the digit the model predicts. 
model.predict(x_test[:4])
