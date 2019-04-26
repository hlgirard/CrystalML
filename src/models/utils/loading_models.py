import pkg_resources
import logging
import re
from tensorflow.keras.models import model_from_json
logging.getLogger('tensorflow').disabled = True

def load_model(model_name):
    '''Loads model from path and get most recent associated weights'''

    model_basename = model_name.split('.')[0]
    model_path = pkg_resources.resource_filename('models', model_name)

    ## Load model from JSON
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    ## Load weights into model
    model_list = sorted([model for model in pkg_resources.resource_listdir('models', '.') if model.startswith(model_basename) and model.endswith('.h5')], key = lambda x: int(re.search(r'\d+', x).group(0)))
    logging.info("Loading model weights: %s", model_list[-1])
    model.load_weights(pkg_resources.resource_filename('models', model_list[-1]))

    return model