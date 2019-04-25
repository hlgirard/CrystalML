import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Required to avoid OMP: Error #15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tensorflow logging level set to ALL (0, default), WARNING (1), ERROR (2) or NONE (3)

import re
import logging

import numpy as np
from tqdm import tqdm
import pandas as pd

from tensorflow.keras.models import model_from_json

from src.data.utils import select_rectangle, get_date_taken, open_grey_scale_image
from src.data.segment_droplets import crop, segment, extract_indiv_droplets
from src.visualization.image_processing_overlay import save_overlay_image
from src.visualization.process_plotting import plot_crystal_data

def load_model(path):
    '''Loads model from path and get most recent associated weights'''

    model_name = path.split('/')[-1].split('.')[0]

    ## Load model from JSON
    with open(path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    ## Load weights into model
    model_list = sorted([model for model in os.listdir("models") if model.startswith(model_name) and model.endswith('.h5')], key = lambda x: int(re.search(r'\d+', x).group(0)))
    logging.info("Loading model weights: %s", model_list[-1])
    model.load_weights("models/" + model_list[-1])

    return model

def process_image(image_path, crop_box, model, save_overlay = False):
    '''
    Process a single image to obtain the number of droplets with and without crystals

    Parameters
    ----------
    imgage_path: string
        Path to the image to process
    crop_box: (minRow, maxRow, minCol, maxCol)
        Cropping box to select the region of interest
    model: tensorflow model
        Instance of a tensorflow model trained to discriminate droples containing crystals vs. clear
    save_overlay: bool, optional
        Save an image with green / red overlays for drops containing crystals / empty to `image_path / overlay`

    Returns
    -------
    (date_take: datetime, num_drops: int, num_clear: int, num_crystal: int)
        Date from the EXIF data, number of drops, number of clear drops, number of drops containing crystals
    '''

    # Open image
    date_taken = get_date_taken(image_path)
    image = open_grey_scale_image(image_path)

    # Crop image
    cropped = crop(image, crop_box)

    # Segment image
    (labeled, _, _) = segment(cropped)

    # Extract individual droplets
    drop_images, regProps = extract_indiv_droplets(cropped, labeled)

    # Predict labels from model
    if drop_images and len(drop_images) > 0:
        X  = np.asarray(drop_images)
        Y = model.predict_classes(X).flatten().tolist()

        num_drops = len(Y)
        num_clear = Y.count(0)
        num_crystal = num_drops - num_clear
    
    else:
        logging.warning("No droplets found in image %s", image_path)
        num_drops = 0
        num_clear = 0
        num_crystal = 0

    # Save overlay if applicable
    if save_overlay:
        path = os.path.join(os.path.dirname(image_path), 'overlay', os.path.basename(image_path))
        save_overlay_image(path, cropped, regProps, Y)

    return (date_taken, num_drops, num_clear, num_crystal)

def process_image_folder(directory, crop_box=None, show_plot=False, save_overlay=False):

    # List images in directory
    image_list = [file for file in os.listdir(directory) if file.endswith('.JPG')]

    # Load model
    model = load_model("models/cnn-simple-model.json")
    
    # Obtain crop box from user if not passed as argument
    if not crop_box:
        first_image = open_grey_scale_image(os.path.join(directory, image_list[0]))
        crop_box = select_rectangle(first_image)


    # Process all images from directory in series
    data = []
    for image_name in tqdm(image_list):
        data.append(process_image(os.path.join(directory, image_name), crop_box, model, save_overlay = save_overlay) + (image_name,)) 

    # Make a dataframe from the data and save it to disk
    df = pd.DataFrame(sorted(data, key = lambda x: x[0]), columns=["DateTime", "Num drops", "Num clear", "Num crystal", "Image Name"])
    df['RelTime'] = (df['DateTime'] - df['DateTime'][0]).dt.total_seconds()
    df.to_csv(os.path.join(directory, "crystalData.csv"))

    # Plot the data for imediate visualization
    if show_plot:
        plot_crystal_data(df, directory)


if __name__ == "__main__":
    folder = "notebooks/example_data"

    process_image_folder(folder, save_overlay = True)
