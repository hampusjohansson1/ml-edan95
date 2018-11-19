from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory="../datasets/flowers_split/test",
    target_size=(150, 150),
    batch_size=100,
    class_mode="categorical")

model = load_model('./flowers.h5')

model.evaluate_generator(test_generator,steps=50,verbose=1)