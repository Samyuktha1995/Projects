import pandas as pd
import numpy as np
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from PIL import Image
from math import sqrt


batch_size=3
epochs=20
img_rows=75
img_cols=75

df = pd.read_csv('trainData.csv')
sdf = pd.read_csv('sample.csv')
df2 = pd.read_csv('trainData.csv')
train_labels = df['DGCI'].values.tolist()
train_labels = np.array(train_labels,dtype=float)
datagen = ImageDataGenerator()
train_generator = datagen.flow_from_dataframe(dataframe=df, directory=".\Archive",
                                              x_col="Id", y_col="DGCI", has_ext=False,
                                              class_mode="other", target_size=(img_rows, img_cols),
                                              color_mode='rgb', batch_size=batch_size)

sample_generator = datagen.flow_from_dataframe(dataframe=sdf, directory=".\Archive",
                                              x_col="Id", y_col="DGCI", has_ext=False,
                                              class_mode="other", target_size=(img_rows, img_cols), batch_size=batch_size)

train_ids = df['Id'].values.tolist()
test_ids = sdf['Id'].values.tolist()
train = []
test = []
for id in train_ids:
    img = Image.open('Archive\\'+str(id)+'.jpg')
    img = img.resize((img_rows,img_cols))
    tmp = np.array(img)
    train.append(tmp)

for id in test_ids:
    img = Image.open('Archive\\'+str(id)+'.jpg')
    img = img.resize((img_rows,img_cols))
    tmp = np.array(img)
    test.append(tmp)



model = Sequential()
model.add(Conv2D(32, (3, 3),padding='same',
                 input_shape=(img_rows, img_cols,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
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
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))


model.compile(optimizer=keras.optimizers.Adadelta(),loss="mean_squared_error", metrics=["mse"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

train=np.array(train,dtype=float)
test=np.array(test,dtype=float)
model.fit(train, train_labels, epochs=epochs, batch_size=64)


               #     validation_data=valid_generator,
               #     validation_steps=STEP_SIZE_VALID,
               #     epochs=10)

scores = model.evaluate(train, train_labels)
print(sqrt(scores[1]))

predict = model.predict(test)
#print(predict)
print("lenght:")
print(len(predict))

#df1 = pd.read_csv('trainData.csv')
sdf['DGCI'] = predict
sdf.to_csv('prediction2.csv',index=False)

prediction = model.predict(train)
#print("length: "+len(prediction))
df2['DGCI'] = prediction
df2.to_csv('dummy.csv',index=False)