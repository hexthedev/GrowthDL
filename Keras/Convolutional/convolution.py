import keras.models
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


batch = 32

epochs = 100
steps = 8000/batch
val_steps = 2000/batch


def build_classifier():
    # Build the fukin CNN
    classifier = Sequential()

    # Convulution
    classifier.add(Convolution2D(32, (3, 3), input_shape=(512, 512, 3), activation="relu"))
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(32, (3, 3), activation="relu"))
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return classifier


# image augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    r'C:\Users\james\Desktop\Convolutional_Neural_Networks\Convolutional_Neural_Networks\dataset\training_set',
    target_size=(512, 512),
    batch_size=batch,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    r'C:\Users\james\Desktop\Convolutional_Neural_Networks\Convolutional_Neural_Networks\dataset\test_set',
    target_size=(512, 512),
    batch_size=batch,
    class_mode='binary'
)

classifier = build_classifier()

classifier.fit_generator(
    training_set,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=val_steps,
    workers=4
)

classifier.save('my_model.h5')