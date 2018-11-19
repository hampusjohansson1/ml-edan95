from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="../datasets/flowers_split/train",
    target_size=(150, 150),
    batch_size=40,
    class_mode="categorical")

validation_generator = validation_datagen.flow_from_directory(
    directory="../datasets/flowers_split/validation",
    target_size=(150, 150),
    batch_size=40,
    class_mode="categorical")

model.compile(loss='categorical_crossentropy',
optimizer='nadam',
metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=90,
    epochs=6,
    validation_data=validation_generator,
    validation_steps=50)

model.save('./aug-flowers.h5')