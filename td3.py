import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.datasets import mnist
import time
from sklearn import metrics


(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
startTime = time.time()
nbClasses = 10
nbNeuronsHL = 200 # nb of neurons on the hidden layer
xTrain = xTrain.reshape(60000, 784)
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
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=20, batch_size=128)
score = model.evaluate(xTest, yTest)
score_train = model.evaluate(xTrain, yTrain)

# confusion matrix
y_predicted = model.predict(xTest)
confusion_matrix = metrics.confusion_matrix(yTest.argmax(axis=1), y_predicted.argmax(axis=1))

print("%s: %.2f%%" % ('accuracy_test', score[1]*100))
print("%s: %.2f%%" % ('test_loss', score[0]*100))
print("%s: %.2f%%" % ('accuracy_train', score_train[1]*100))
print("%s: %.2f%%" % ('train_loss', score_train[0]*100))
print('Test time: {0}'.format(time.time() - startTime))
print('Confusion_matrix')
print(confusion_matrix)
