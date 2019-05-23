import pkg_resources
import logging
import re
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Required to avoid OMP: Error #15
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras import backend as K
K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

def load_model(model_name):
    '''Loads model from path and get most recent associated weights'''

    model_basename = model_name.split('.')[0]
    model_path = pkg_resources.resource_filename('models', model_name)

    ## Load model from JSON
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = tf.keras.models.model_from_json(loaded_model_json)

    ## Load weights into model
    model_list = sorted([model for model in pkg_resources.resource_listdir('models', '.') if model.startswith(model_basename) and model.endswith('.h5')], key=lambda x: int(re.search(r'\d+', x).group(0)))
    logging.info("Loading model weights: %s", model_list[-1])
    model.load_weights(pkg_resources.resource_filename('models', model_list[-1]))

    return model