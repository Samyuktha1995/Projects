import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import numpy as np


batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#x_train = tf.reshape(x_train, [-1, 28, 28, 1])
#x_test = tf.reshape(x_test, [-1, 28, 28, 1])
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
          #validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test = pd.read_csv('mnist_test.csv')
test = test.iloc[:,1:].values
test = test.reshape(test.shape[0], img_rows, img_cols, 1)
Y_pred_test = model.predict(test)
#Y_pred_test = np.argmax(Y_pred_test, axis=0)

correct_predictions = []
for one_hot in Y_pred_test:
    correct_predictions.append(np.argmax(one_hot))


"""
1. Read mnist_test and put it into a panda df
2. Convert this into a numpy matrix
3. Predict the solutions for this numpy matrix.
4. Read the mnist sample file
5. Replace the category column with the predictions
6. Write
7. Profit!
"""
df = pd.read_csv('mnist_sample.csv') #fill this up with the sample_data part
df['Category'] = correct_predictions
df.index.name = 'Id'
df.to_csv('submission.csv', index=False)

