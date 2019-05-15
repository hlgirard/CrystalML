import os

import math
import logging
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.utils import select_rectangle, get_date_taken, open_grey_scale_image
from src.data.segment_droplets import crop, segment, extract_indiv_droplets
from src.visualization.image_processing_overlay import save_overlay_image
from src.visualization.process_plotting import plot_crystal_data
from src.models.utils.loading_models import load_model

def process_image(image_path, crop_box, model, save_overlay=False):
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
    (labeled, _) = segment(cropped)

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

def process_image_batch(image_list, crop_box, model_name, batch_number=0, verbosity=logging.WARNING, save_overlay=False):
    '''Process a batch of images and return a list of results

    Parameters
    ----------
    image_list: list[string]
        List of paths to the image to process
    crop_box: (minRow, maxRow, minCol, maxCol)
        Cropping box to select the region of interest
    model_name: string
        Path to the tensorflow model to load
    batch_numer: int
        Batch number when parallel processing for progress bar display. Default: 0
    verbosity: int
        verbosity as a logging level. Default: logging.WARNING (30)
    save_overlay: bool, optional
        Save an image with green / red overlays for drops containing crystals / empty to `image_path / overlay`. Default: False

    Returns
    -------
    list[(date_take: datetime, num_drops: int, num_clear: int, num_crystal: int, image_name: string)]
        List of extracted parameters for each of the images
        Date from the EXIF data, number of drops, number of clear drops, number of drops containing crystals, name of the image
    '''

    # Instantiate the model
    model = load_model(model_name)

    # Process the data
    data = []
    for image_path in tqdm(image_list,
                           desc="Thread #{}".format(batch_number),
                           position=batch_number,
                           leave=(batch_number == 0),
                           disable=((batch_number != 0) and (verbosity > 20))
                           ):
        image_name = os.path.basename(image_path)
        data.append(process_image(image_path, crop_box, model, save_overlay=save_overlay) + (image_name,))

    return data

def process_image_folder(directory, crop_box=None, show_plot=False, save_overlay=False, show_segmentation=False):

    # List images in directory
    image_list = [os.path.join(directory, image_path) for image_path in os.listdir(directory) if image_path.endswith('.JPG')]

    # Define the model path
    model_name = "cnn-simple-model.json"

    # Obtain crop box from user if not passed as argument
    while not crop_box:
        logging.info("Crop box not passed, opening ROI selection tool")
        first_image = open_grey_scale_image(image_list[0])
        crop_box = select_rectangle(first_image)

    # If show segmentation flag is asserted, display the segmentation of an image
    if show_segmentation:
        idx_80 = int(len(image_list) * 0.8)
        image_80 = crop(open_grey_scale_image(image_list[idx_80]), crop_box)
        logging.info("Segmentation check requested. Segmenting image %s", image_list[idx_80])
        labeled, _ = segment(image_80)

        from skimage.color import label2rgb
        overlay_image = label2rgb(labeled, image=image_80, bg_label=1)

        import matplotlib
        matplotlib.use('Qt4Agg', force=True)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.set_tight_layout(True)
        plt.ion()
        plt.imshow(overlay_image)
        plt.show(block=False)
        plt.pause(0.1)

        result = input("Continue? (y/n)\n")

        while result != 'y':
            if result == 'n':
                plt.close()
                logging.warning("Segmentation rejected. Stopping the application")
                return
            else:
                result = input("Please press 'y' for yes or 'n' for no\n")

        plt.close()

    # Compute the number of batches necessary
    num_images = len(image_list)
    logging.info("Number of images: %d", num_images)
    num_cpu = os.cpu_count()

    if num_images < num_cpu:
        logging.info("Processing on a single thread.")
        flat_data = process_image_batch(image_list, crop_box, model_name, 0, logging.root.level, save_overlay)
    else:
        logging.info("Processing in parallel")
        batch_size = max([1, num_images // (num_cpu-1)])
        logging.info("Batch size: %d", batch_size)
        num_batches = int(math.ceil(num_images // batch_size))
        logging.info("Number of batches: %d", num_batches)

        # Process all images from directory in parallel
        data = Parallel(n_jobs=-2)(delayed(process_image_batch)(image_list[i*batch_size:min([(i+1)*batch_size, num_images])],
                                                                        crop_box,
                                                                        model_name,
                                                                        i,
                                                                        logging.root.level,
                                                                        save_overlay)
                                    for i in range(num_batches))

        flat_data = [item for sublist in data for item in sublist]

    # Make a dataframe from the data and save it to disk
    df = pd.DataFrame(sorted(flat_data, key=lambda x: x[0]), columns=["DateTime", "Num drops", "Num clear", "Num crystal", "Image Name"])
    df['RelTime'] = (df['DateTime'] - df['DateTime'][0]).dt.total_seconds()
    df.to_csv(os.path.join(directory, "crystalData.csv"))
    logging.info("Saved data to disk.")

    # Plot the data for imediate visualization
    if show_plot:
        plot_crystal_data(df, directory)