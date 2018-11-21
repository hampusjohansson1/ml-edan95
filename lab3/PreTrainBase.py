from keras.applications import InceptionV3
from keras.models import load_model
from keras import utils
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.inception_v3 import preprocess_input

conv_base = InceptionV3(weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))

conv_base.trainable = False
#print(conv_base.summary())

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True

base_dir = '/Users/hampus/Documents/Code/tillämpadMl/datasets/flowers_split'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

load=False

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=40,
    class_mode='categorical')

validation_generator = validate_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=15,
    class_mode='categorical')

def generateModel():

    #Create a dense model
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer="nadam",
    loss='categorical_crossentropy',
    metrics=['acc'])

    #Run the training
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=90,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=100)

    model.save('./pretrained2.h5')

    return model

if load:
    model = load_model('./pretrained2.h5')
    print(model.summary())
else:
    model = generateModel()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator,steps=50)
print('test acc:', test_acc)
print('test loss:', test_loss)

Y_pred = model.predict_generator(test_generator, 1500 // 21)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes[:y_pred.size], y_pred))
