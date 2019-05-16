from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

from time import time
import pkg_resources
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train_cnn_simple_from_directory(training_directory, bTensorboard):

        logging.info("Starting training of simple CNN from directory %s", training_directory)

        ## Define the model
        model = Sequential()

        # 1st layer convolutional
        model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2nd layer convolutional
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 3rd layer convolutional
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten
        model.add(Flatten())

        # 4th layer fully connected
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # 5th layer fully connected
        model.add(Dense(1))
        model.add(Activation('sigmoid'))


        model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

        ## Prepare the data
        batch_size = 16

        # this is the augmentation configuration we will use
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2)


        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                training_directory,  # this is the target directory
                target_size=(150, 150),  # all images will be resized to 150x225
                batch_size=batch_size,
                color_mode ='grayscale',
                class_mode='binary', # since we use binary_crossentropy loss, we need binary labels
                subset='training')

        # this is a similar generator, for validation data
        validation_generator = train_datagen.flow_from_directory(
                training_directory,
                target_size=(150, 150),
                batch_size=batch_size,
                color_mode='grayscale',
                class_mode='binary',
                subset='validation')

        model.summary()

        if bTensorboard:
                log_dir = "logs/{}".format(time())
                logging.info("Saving tensorboard loggs to: %s", log_dir)
                tensorboard = TensorBoard(log_dir=log_dir)

        training_history = model.fit_generator(
                train_generator,
                steps_per_epoch=2000 // batch_size,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=800 // batch_size,
                callbacks=[tensorboard] if tensorboard else None)

        # Save Model to JSON
        model_json = model.to_json()
        model_path = pkg_resources.resource_filename('models', "cnn-simple-model.json")
        logging.info("Saving model JSON file to: %s", model_path)
        with open(model_path, "w") as json_file:
                json_file.write(model_json)
        # Save weigths
        model_weights_path = pkg_resources.resource_filename('models', "cnn-simple-model-{}.h5".format(time()))
        logging.info("Saving model weights to %s", model_weights_path)
        model.save_weights(model_weights_path)