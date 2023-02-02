# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 18:16:47 2022

@author: khalid
"""

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation


"""Loading 'MNIST Data Set' : Training samples = 60000, Testing Samples = 10000"""
from keras.datasets import mnist
"""After loading the MNIST data, Divide into train and Test datasets"""
(Xtrain,Ytrain),(Xtest,Ytest) = mnist.load_data()
"""Size Xtrain"""
#print(Xtrain.shape)
"""Size Xtest"""
#print(Xtest.shape)




"""Affichage"""
plt.imshow(Xtrain[0],cmap=plt.cm.binary)
#plt.show()
"""Checking the values of each pixel Before Normalization"""
print(Xtrain[0])
"""Normalizing the data"""
Xtrain = Xtrain / 255
Xtest = Xtest / 255
"""Checking the values of each pixel After Normalization"""
print(Xtrain[0])



"""Resizing imag to make it suitable for apply Convolution operation"""
IMG_SIZE = 28
Xtrain = np.array(Xtrain).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Xtest = np.array(Xtest).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print("Training Samples dimension : ",Xtrain.shape)
print("Testing Samples dimension : ",Xtest.shape)
"""Creating a Deep Neural Network"""
model = Sequential()
#First convolution layer  28-3+1 = 26*26
model.add(Conv2D(64,(3,3),input_shape=Xtrain.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#2nd convolution layer   
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#3nd convolution layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Fully connected layer 1
model.add(Flatten()) #before using fully connected layer, need to be flatten so that 2D to 1D
model.add(Dense(64))
model.add(Activation("relu"))

#Fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

#Fully connected layer 3
model.add(Dense(10))  # 0-9
model.add(Activation("softmax")) #class probabilité

"""Summary model"""
model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

"""Training my model"""
model.fit(Xtrain,Ytrain,epochs=10,validation_split=0.3)
"""Evaluateing on testing data set MNIST"""
test_loss, test_acc = model.evaluate(Xtest,Ytest)
print("Test loss on 10,000 test samples  ",test_loss)
print("validation Accuracy on 10,000 test semples ",test_acc)

model.save("model.h5")

predictions = model.predict(Xtest)

print(np.argmax(predictions[0]))



