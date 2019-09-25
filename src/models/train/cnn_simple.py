from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.constraints import unit_norm, max_norm
from tensorflow.keras.optimizers import SGD, Adadelta
import tensorflow.keras.backend as K

from time import time
import pkg_resources
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class cnn_simple:
        @staticmethod
        def build(width, height, depth):

                model = Sequential()
                inputShape = (height, width, depth)
 
                # if we are using "channels first", update the input shape
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)

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
                model.add(Dense(1, activation='sigmoid'))

                return model

class cnn_regularized:
        '''
        Regularization of the weights is added to each layer.
        No measurable improvement in accuracy but significantly longer run time.
        '''
        @staticmethod
        def build(width, height, depth):

                model = Sequential()
                inputShape = (height, width, depth)
 
                # if we are using "channels first", update the input shape
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)

                # 1st layer convolutional
                model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                # 2nd layer convolutional
                model.add(Conv2D(32, (3, 3), activation='relu', kernel_constraint=max_norm(3., axis=[0, 1, 2])))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                # 3rd layer convolutional
                model.add(Conv2D(64, (3, 3), activation='relu', kernel_constraint=max_norm(3., axis=[0, 1, 2])))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                # Flatten
                model.add(Flatten())

                # 4th layer fully connected
                model.add(Dense(64, activation='relu', kernel_constraint=unit_norm()))
                model.add(Dropout(0.5))

                # 5th layer fully connected
                model.add(Dense(1, activation='sigmoid'))

                return model

class cnn_dropout:
        '''
        Dropouts added to each layer to prevent overfitting.
        Some improvement in testing loss but significantly longer run time.
        '''
        @staticmethod
        def build(width, height, depth):

                model = Sequential()
                inputShape = (height, width, depth)
 
                # if we are using "channels first", update the input shape
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)

                # 1st layer convolutional
                model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1), activation='relu'))
                model.add(Dropout(0.2))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                # 2nd layer convolutional
                model.add(Conv2D(32, (3, 3), activation='relu'))
                model.add(Dropout(0.2))

                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                # 3rd layer convolutional
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(Dropout(0.2))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                # Flatten
                model.add(Flatten())

                # 4th layer fully connected
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.5))

                # 5th layer fully connected
                model.add(Dense(1, activation='sigmoid'))

                return model

def train_cnn_simple_from_directory(training_directory, bTensorboard):

        logging.info("Starting training of simple CNN from directory %s", training_directory)

        ## Define the model
        model = cnn_simple.build(150, 150, 1)

        #optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

        model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

        ## Prepare the data
        batch_size = 16
        num_samples = sum([len(os.listdir(os.path.join(training_directory, categoryDir))) for categoryDir in os.listdir(training_directory) if os.path.isdir(os.path.join(training_directory, categoryDir))])
        num_training = int(0.8 * num_samples)
        num_validation = num_samples - num_training

        logging.info("Number of labeled images: %d", num_samples)
        logging.info("Number of training samples: %d", num_training)
        logging.info("Number of validation samples: %d", num_validation)

        # this is the augmentation configuration we will use
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2)


        # Training generator
        train_generator = train_datagen.flow_from_directory(
                training_directory,
                target_size=(150, 150),  # all images will be resized to 150x150
                batch_size=batch_size,
                color_mode='grayscale',
                class_mode='binary',
                subset='training')

        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
                training_directory,
                target_size=(150, 150),
                batch_size=batch_size,
                color_mode='grayscale',
                class_mode='binary',
                subset='validation',
                shuffle=False)

        model.summary()

        if bTensorboard:
                log_dir = "logs/{}".format("cnn_simple_" + str(time()))
                logging.info("Saving tensorboard loggs to: %s", log_dir)
                callbacks = [TensorBoard(log_dir=log_dir)]
        else:
                callbacks = None

        training_history = model.fit_generator(
                train_generator,
                steps_per_epoch=num_training // batch_size,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=num_validation // batch_size,
                callbacks=callbacks
        )

        # Save Model to JSON
        model_json = model.to_json()
        model_path = pkg_resources.resource_filename('models', "cnn-simple-model.json")
        logging.info("Saving model JSON file to: %s", model_path)
        with open(model_path, "w") as json_file:
                json_file.write(model_json)
        
        # Save weights
        model_weights_path = pkg_resources.resource_filename('models', "cnn-simple-model-{}.h5".format(time()))
        logging.info("Saving model weights to %s", model_weights_path)
        model.save_weights(model_weights_path)

        # Display confusion matrix
        try:
                from sklearn.metrics import classification_report, confusion_matrix
                import numpy as np
                Y_pred = model.predict_generator(validation_generator, num_validation // batch_size+1)
                y_pred = np.argmax(Y_pred, axis=1)
                print('-------- Confusion Matrix --------')
                print(confusion_matrix(validation_generator.classes, y_pred))
                print('-------- Classification Report --------')
                target_names = ['Clear', 'Crystal']
                print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
        except ImportError:
                logging.info("sklearn is required to print confusion matrix and classification report.")