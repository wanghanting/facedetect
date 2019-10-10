import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.datasets import mnist

# #load dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
(xTrain,yTrain),(xTest,yTest) = mnist.load_data()
nbClasses = 10
nbNeuronsHL = 20 # nb of neurons on the hidden layer xTrain = xTrain.reshape(60000, 784)
xTest = xTest.reshape(10000, 784)
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
yTrain = keras.utils.to_categorical(yTrain, nbClasses)
yTest = keras.utils.to_categorical(yTest, nbClasses)
model = Sequential()
model.add(Dense(nbNeuronsHL, input_dim=784, activation='sigmoid'))
model.add(Dense(nbClasses, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=20, batch_size=128)
score = model.evaluate(xTest,yTest)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
