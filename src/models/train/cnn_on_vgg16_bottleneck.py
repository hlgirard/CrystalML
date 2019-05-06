import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD

# dimensions of our images.
img_width, img_height = 250, 150

top_model_weights_path = 'models/bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 224
nb_train_crystal = 117
nb_train_clear = 107
nb_validation_samples = 64
nb_validation_crystal = 29
nb_validation_clear = 35
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    print("Loading VGG16 model")
    model_vgg16 = applications.VGG16(include_top=False, weights='imagenet')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model_vgg16.compile(optimizer=sgd, loss='categorical_crossentropy')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model_vgg16.predict_generator(generator, nb_train_samples // batch_size)
    with open('models/bottleneck_features_train.npy', 'wb') as f:
        np.save(f, bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model_vgg16.predict_generator(generator, nb_validation_samples // batch_size)
    with open('models/bottleneck_features_validation.npy', 'wb') as f:
        np.save(f, bottleneck_features_validation)


def train_top_model():
    with open('models/bottleneck_features_train.npy', 'rb') as f:
        train_data = np.load(f)
    train_labels = np.array([0] * nb_train_clear + [1] * nb_train_crystal)

    with open('models/bottleneck_features_validation.npy', 'rb') as f:
        validation_data = np.load(f)
    validation_labels = np.array([0] * nb_validation_clear + [1] * nb_validation_crystal)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    training_history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    return training_history


save_bottlebeck_features()
history = train_top_model()
