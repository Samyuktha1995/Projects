import pandas as pd
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from PIL import Image
from math import  sqrt


batch_size=3
epochs=15
img_rows=32
img_cols=32

df = pd.read_csv('trainData.csv')
sdf = pd.read_csv('sample.csv')
datagen = ImageDataGenerator()
train_generator = datagen.flow_from_dataframe(dataframe=df, directory=".\Archive",
                                              x_col="Id", y_col="DGCI", has_ext=False,
                                              class_mode="other", target_size=(img_rows, img_cols),
                                              color_mode='rgb', batch_size=batch_size)

sample_generator = datagen.flow_from_dataframe(dataframe=sdf, directory=".\Archive",
                                              x_col="Id", y_col="DGCI", has_ext=False,
                                              class_mode="other", target_size=(img_rows, img_cols), batch_size=batch_size)

label_map = train_generator.classes
print(label_map)


model = Sequential()

model.add(Conv2D(32, (2, 2),padding='same',
                 input_shape=(img_rows, img_cols,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))


model.compile(optimizer=keras.optimizers.Adadelta(),loss="mean_squared_error", metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=10,
                    epochs=epochs)


               #     validation_data=valid_generator,
               #     validation_steps=STEP_SIZE_VALID,
               #     epochs=10)

scores = model.evaluate_generator(generator=train_generator, steps=len(train_generator),verbose=0)
print(sqrt(scores[0]))

predict = model.predict_generator(generator=train_generator,steps=len(train_generator),verbose=0)
#print(predict)
print("lenght:")
print(len(predict))

df1 = pd.read_csv('trainData.csv')
df1['DGCI'] = predict
df1.to_csv('prediction1.csv',index=False)