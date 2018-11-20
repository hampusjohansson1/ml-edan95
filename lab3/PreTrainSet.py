from keras.applications import InceptionV3
from keras.applications.inception_v3 import decode_predictions
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.inception_v3 import preprocess_input

conv_base = InceptionV3(weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))

base_dir = '/Users/hampus/Documents/Code/tillämpadMl/datasets/flowers_split'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

def extract_features(directory, count):
    features = np.zeros(shape=(count, 3, 3, 2048))
    labels = np.zeros(shape=(count,5))

    generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')

    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1
        if i * batch_size >= count:
            break
    return features, labels

#Extract features for training and validation
train_features, train_labels = extract_features(train_dir, 3640)
validation_features, validation_labels = extract_features(validation_dir, 1540)

train_features = np.reshape(train_features, (3640, 3*3*2048))
validation_features = np.reshape(validation_features, (1540, 3*3*2048))

#Create a dense model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=3*3*2048))
model.add(layers.Dense(5, activation='softmax'))
model.compile(optimizer="nadam",
loss='categorical_crossentropy',
metrics=['acc'])

#Run the training
history = model.fit(train_features, train_labels,
epochs=5,
batch_size=20,
validation_data=(validation_features, validation_labels))

#Test the network
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50,input_shape=(150, 150, 3))
print('test acc: ', test_acc)
print('test_loss: ', test_loss)

Y_pred = model.predict_generator(test_generator, steps=50,input_shape=(150, 150, 3))
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))