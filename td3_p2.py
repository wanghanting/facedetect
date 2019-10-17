import keras as K
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import fashion_mnist
import time
from sklearn import metrics
import matplotlib.pyplot as plt


(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()
# to calculate time
startTime = time.time()
nbClasses = 10
# nb of neurons on the hidden layer
nbNeuronsHL = 200
xTrain = xTrain.reshape(60000, 784)
xTest = xTest.reshape(10000, 784)
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
yTrain = K.utils.to_categorical(yTrain, nbClasses)
yTest = K.utils.to_categorical(yTest, nbClasses)

model = Sequential()
model.add(Dense(nbNeuronsHL, input_dim=784, activation='sigmoid'))
model.add(Dense(nbClasses, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=30, batch_size=128)
score = model.evaluate(xTest, yTest)
score_train = model.evaluate(xTrain, yTrain)

# confusion matrix
y_predicted = model.predict(xTest)
confusion_matrix = metrics.confusion_matrix(yTest.argmax(axis=1), y_predicted.argmax(axis=1))

# validation
history = model.fit(xTrain, yTrain, validation_split=0.25, epochs=30, batch_size=16, verbose=1)

# visualization
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print("%s: %.2f%%" % ('accuracy_test', score[1]*100))
print("%s: %.2f%%" % ('test_loss', score[0]*100))
print("%s: %.2f%%" % ('accuracy_train', score_train[1]*100))
print("%s: %.2f%%" % ('train_loss', score_train[0]*100))
print('Test time: {0}'.format(time.time() - startTime))
print('Confusion_matrix')
print(confusion_matrix)
