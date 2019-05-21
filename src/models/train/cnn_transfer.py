from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.constraints import unit_norm, max_norm
from tensorflow.keras.optimizers import SGD, Adadelta
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras.backend as K

from time import time
import pkg_resources
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class cnn_transfer_ResNet50:
    @staticmethod
    def build(width, height, top_layer = -1):

        # Load resnet model
        resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(width, height, 3))

        # Get input and bottleneck layers
        bottleneck_input  = resnet_model.get_layer(index=0).input
        bottleneck_output = resnet_model.get_layer(index=top_layer).output
        bottleneck_model  = Model(inputs=bottleneck_input, outputs=bottleneck_output)

        # Freeze all layers of the pretrained model
        for layer in bottleneck_model.layers:
            layer.trainable = False

        # Make a new model on top
        model = Sequential()
        model.add(bottleneck_model)
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        return model

def train_cnn_transfer_from_directory(training_directory, bTensorboard, last_layer=-1):

    model = cnn_transfer_ResNet50.build(150, 150, last_layer)

    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
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

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            training_directory,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            color_mode='rgb',
            class_mode='binary', # since we use binary_crossentropy loss, we need binary labels
            subset='training')

    # this is a similar generator, for validation data
    validation_generator = train_datagen.flow_from_directory(
            training_directory,
            target_size=(150, 150),
            batch_size=batch_size,
            color_mode='rgb',
            class_mode='binary',
            subset='validation')

    model.summary()

    if bTensorboard:
        log_dir = "logs/{}".format("cnn_onResnet_" + str(time()))
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
    model_path = pkg_resources.resource_filename('models', "cnn-onResnet-model.json")
    logging.info("Saving model JSON file to: %s", model_path)
    with open(model_path, "w") as json_file:
            json_file.write(model_json)

    # Save weigths
    model_weights_path = pkg_resources.resource_filename('models', "cnn-onResnet-model-{}.h5".format(time()))
    logging.info("Saving model weights to %s", model_weights_path)
    model.save_weights(model_weights_path)