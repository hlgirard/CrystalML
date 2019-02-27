from math import sqrt

import numpy as np

import skimage
from skimage import io, exposure, img_as_float, img_as_ubyte, morphology, filters, util
from skimage.color import rgb2gray, label2rgb
from skimage.feature import canny, blob_dog, blob_log, blob_doh,  peak_local_max
from skimage.filters import sobel, threshold_otsu, try_all_threshold, threshold_local, threshold_minimum
from skimage.segmentation import clear_border, watershed, random_walker
from skimage.measure import label, regionprops
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import closing

from scipy import ndimage as ndi

import os

def open_grey_scale_image(path):
    '''Opens an image and converts it to ubyte and greyscale'''
    f = io.imread(path)
    return img_as_ubyte(rgb2gray(f))

def crop(img, cBox):
    '''Returns a cropped image for cBox = (minRow, maxRow, minCol, maxCol)'''
    (minRow, maxRow, minCol, maxCol) = cBox
    return img[minRow:maxRow, minCol:maxCol]

def segment(img, mask_val = 0.2):
    '''
    Segments droplets in an image using a watershed algorithm.

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of an image cropped to show only the droplets region
    mask_val: float, optional
        Masking value (0-1) for the distance plot to remove small regions. Default 0.2

    Returns
    -------
    numpy.ndarray
        labeled array of the same shape as input image where each region is assigned a disctinct integer label.
    '''

    # Adaptive equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Minimum threshold
    binary = img_adapteq > threshold_minimum(img_adapteq)

    # Remove dark spots and connect bright spots
    closed = closing(binary)

    # Calculate the distance to the dark background
    distance = ndi.distance_transform_edt(closed)

    # Increase contrast of the the distance image
    cont_stretch = exposure.rescale_intensity(distance, in_range='image')

    # Mask the distance image to remove interstitial points
    masked = cont_stretch.copy()
    masked[masked < mask_val] = 0

    # Find local maximas of the distance image
    local_maxi = peak_local_max(masked, indices=False, min_distance=3, threshold_abs=0.6)

    # Markers for watershed are the local maxima of the distance image
    markers, _ = ndi.label(local_maxi)

    # Run watershed algorithm on the inverse of the distance image
    #segmented = watershed(-cont_stretch, markers, mask = image, watershed_line=True)
    segmented = watershed(-masked, markers, mask = masked > 0)

    # Label the segments of the image
    labeled, _ = ndi.label(segmented)

    return labeled

def extract_indiv_droplets(img, labeled, border = 25, ecc_cutoff = 0.65):
    '''
    Separate the individual droplets as their own image.

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of an image cropped to show only the droplets region
    labeled: numpy.ndarray
        Label array corresponding to 'img' where each region is assigned a disctinct integer value
    border: int, optional
        Number of pixels to add on each side of the labeled area to produce the final image. Default 25
    ecc_cutoff: float, optional
        Maximum eccentricity value of the labeled region. Regions with higher eccentricity will be ignored. Default 0.65

    Returns
    -------
    numpy.ndarray
        labeled array of the same shape as input image where each region is assigned a disctinct integer label.
    '''

    # Get region props
    reg = regionprops(labeled, coordinates='rc')

    # Initialize list of images
    img_list = []

    for region in reg:
        if region.eccentricity < ecc_cutoff:
            (min_row, min_col, max_row, max_col) = region.bbox
            img_list.append(img[min_row-border:max_row+border,min_col-border:max_col+border])

    return img_list


