import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Required to avoid OMP: Error #15
import re
import logging

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from plotly import offline as py
from plotly import tools
import plotly.graph_objs as go
import plotly.io as pio
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.models import model_from_json

from src.data.utils import select_rectangle, get_date_taken
from src.data.segment_droplets import open_grey_scale_image, crop, segment, extract_indiv_droplets

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def load_model(path):
    '''Loads model from path and get most recent associated weights'''

    model_name = path.split('/')[-1].split('.')[0]

    ## Load model from JSON
    with open(path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    ## Load weights into model
    model_list = sorted([model for model in os.listdir("models") if model.startswith(model_name) and model.endswith('.h5')], key = lambda x : int(re.search(r'\d+',x).group(0)))
    logging.info("Loading model weights: {}".format(model_list[-1]))
    model.load_weights("models/" + model_list[-1])

    return model

def process_image(image_path, crop_box, model):

    # Open image
    date_taken = get_date_taken(image_path)
    image = open_grey_scale_image(image_path)

    # Crop image
    cropped = crop(image, crop_box)

    # Segment image
    (labeled, _, _) = segment(cropped)

    # Extract individual droplets
    drop_images = extract_indiv_droplets(cropped, labeled)

    # Predict labels from model
    if drop_images and len(drop_images) > 0:
        X  = np.asarray(drop_images)
        Y = model.predict_classes(X).flatten().tolist()

        num_drops = len(Y)
        num_clear = Y.count(0)
        num_crystal = num_drops - num_clear
    
    else:
        logging.warning("No droplets found in image {}".format(image_path))
        num_drops = 0
        num_clear = 0
        num_crystal = 0

    return (date_taken, num_drops, num_clear, num_crystal)

def plot_crystal_data(df, directory):
    '''Plot data from crystallization experiment'''

    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Crystallization kinetics', 'Process control'))

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num drops"],
        name = 'Total',
        #hoverinfo = 'text',
        #text = df["Image Name"]
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num clear"],
        name = 'Clear'
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num crystal"],
        name = 'Crystal'
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num clear"] / df["Num drops"],
        name = 'Clear/Total'
    ), 1, 1)

    fig['layout']['xaxis'].update(title='Time (s)')
    fig['layout']['yaxis2'].update(title='Number of drops')
    fig['layout']['yaxis1'].update(title='Clear/Total', range=[0,1])

    fig['layout'].update(
        title='Crystallization kinetics data for {}'.format(directory)
    )

    py.plot(fig)
    pio.write_image(fig, os.path.join(directory, 'Crystallization_kinetics_plot.pdf'))

def process_image_folder(directory, crop_box = None):

    # List images in directory
    image_list = [file for file in os.listdir(directory) if file.endswith('.JPG')]
    logging.debug("Image list: {}".format(image_list))

    # Load model
    model = load_model("models/cnn-simple-model.json")
    
    # Obtain crop box from user if not passed as argument
    if not crop_box:
        first_image = open_grey_scale_image(os.path.join(directory, image_list[0]))
        crop_box = select_rectangle(first_image)

    logging.debug("Crop box: {}".format(crop_box))

    # Process all images from directory in series
    data = []
    for image_name in tqdm(image_list):
        logging.debug("Processing image: {}".format(image_name))
        data.append(process_image(os.path.join(directory, image_name), crop_box, model) + image_name) 

    # Make a dataframe from the data and save it to disk
    df = pd.DataFrame(sorted(data, key = lambda x: x[0]), columns=["DateTime", "Num drops", "Num clear", "Num crystal", "Image Name"])
    df['RelTime'] = (df['DateTime'] - df['DateTime'][0]).dt.total_seconds()
    df.to_csv(os.path.join(directory, "crystalData.csv"))

    # Plot the data for imediate visualization
    plot_crystal_data(df, directory)


if __name__ == "__main__":
    folder = "notebooks/example_data"

    process_image_folder(folder)
    




