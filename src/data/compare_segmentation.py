'''
Compare different segmentation implementations.
'''

import os
import logging
from time import time

from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
import matplotlib
matplotlib.use('Qt4Agg', force=True)
import matplotlib.pyplot as plt

from src.data.utils import open_grey_scale_image, select_rectangle, crop
from src.data.segment_droplets import segment, segment_skimage

def segmentation_compare(image_path, crop_box=None):
    '''
    Compares droplet segmentation algorithm implemented with scikit-image and CV2

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of an image cropped to show only the droplets region
    '''

    if not os.path.isfile(image_path):
        raise ValueError("Must provide path to a single image.")

    img = open_grey_scale_image(image_path)

    # Get the crop box from the first image if not provided
    logging.info('Getting crop box from image {}'.format(os.path.basename(image_path)))
    if not crop_box:
        crop_box = select_rectangle(img)

    # Crop image
    cropped = crop(img, crop_box)

    # Segment image scikit-image
    logging.info("Starting scikit-image segmentation")
    t_0_ski = time()
    (labeled_ski, num_reg_ski) = segment_skimage(cropped)
    time_ski = time() - t_0_ski
    logging.info("Finished scikit-image segmentation")

    # Segment image openCV
    logging.info("Starting OpenCV segmentation")
    t_0_cv = time()
    (labeled_cv, num_reg_cv) = segment(cropped)
    time_cv = time() - t_0_cv
    logging.info("Finished OpenCV segmentation")

    # Print report
    print(f"Scikit-image - Droplets: {num_reg_ski} - Time: {time_ski}s")
    print(f"OpenCV       - Droplets: {num_reg_cv} - Time: {time_cv}s")

    # Show segmented images

    img_contrast = equalize_adapthist(cropped, clip_limit=0.06)

    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    ax = axes.ravel()

    ax[0].set_title('Skimage - Contour')
    ax[0].imshow(img_contrast, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].contour(labeled_ski, [0.5], linewidths=1, colors='r')
    ax[1].set_title('Skimage - Overlay')
    ax[1].imshow(label2rgb(labeled_ski, image=img_contrast, bg_label=0), cmap=plt.cm.gray, interpolation='nearest')
    ax[2].set_title('OpenCV - Contour')
    ax[2].imshow(img_contrast, cmap=plt.cm.gray, interpolation='nearest')
    ax[2].contour(labeled_cv, [0.5], linewidths=1, colors='r')
    ax[3].set_title('OpenCV - Overlay')
    ax[3].imshow(label2rgb(labeled_cv, image=img_contrast, bg_label=1), cmap=plt.cm.gray, interpolation='nearest')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show(block=False)

    input("Press any key to exit...")

    plt.close()

    return