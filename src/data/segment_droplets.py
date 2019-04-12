from math import sqrt

import numpy as np

import skimage
from skimage import io, exposure, img_as_ubyte
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import threshold_minimum
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, remove_small_holes, disk

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

def segment(img, exp_clip_limit = 0.06, closing_disk_radius = 2, rm_holes_area = 2048, minima_minDist = 100, mask_val = 0.15):
    '''
    Segments droplets in an image using a watershed algorithm.

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of an image cropped to show only the droplets region
    exp_clip_limit: float [0-1], optional
        clip_limit parameter for adaptive equalisation
    closing_disk_radius: int, optional
        diamater of selection disk for the closing function
    rm_holes_area: int, optional
        maximum area of holes to remove
    minima_minDist: int, optional
        minimum distance between peaks in local minima determination
    mask_val: float, optional
        Masking value (0-1) for the distance plot to remove small regions. Default 0.2

    Returns
    -------
    (labeled: numpy.ndarray, num_maxima: int, num_regions: int)
        labeled: labeled array of the same shape as input image where each region is assigned a disctinct integer label.
        num_maxima: Number of maxima detected from the distance transform
        num_regions: number of labeled regions
    '''

    # Adaptive equalization
    img_adapteq = equalize_adapthist(img, clip_limit = exp_clip_limit)

    # Minimum threshold
    binary = img_adapteq > threshold_minimum(img_adapteq)

    # Remove dark spots and connect bright spots
    closed = binary_closing(binary, selem=disk(closing_disk_radius))
    rm_holes_closed = remove_small_holes(closed, area_threshold=rm_holes_area, connectivity=1)

    # Calculate the distance to the dark background
    distance = ndi.distance_transform_edt(rm_holes_closed)

    # Increase contrast of the the distance image
    cont_stretch = exposure.rescale_intensity(distance, in_range='image')

    # Mask the distance image to remove interstitial points
    masked = cont_stretch.copy()
    masked[masked < mask_val] = 0

    # Find local maximas of the distance image
    local_maxi = peak_local_max(masked, indices=False, min_distance=minima_minDist, threshold_abs=0.6)

    # Markers for watershed are the local maxima of the distance image
    markers, num_maxima = ndi.label(local_maxi)

    # Run watershed algorithm on the inverse of the distance image
    segmented = watershed(-masked, markers, mask = masked > 0)

    # Label the segments of the image
    labeled, num_regions = ndi.label(segmented)

    return (labeled, num_maxima, num_regions)

def extract_indiv_droplets(img, labeled, border = 25, ecc_cutoff = 0.75):
    '''
    Separate the individual droplets as their own image.

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of the segmented image.
    labeled: numpy.ndarray
        Label array corresponding to 'img' where each region is assigned a disctinct integer value
    border: int, optional
        Number of pixels to add on each side of the labeled area to produce the final image.
    ecc_cutoff: float, optional
        Maximum eccentricity value of the labeled region. Regions with higher eccentricity will be ignored.

    Returns
    -------
    list(numpy.ndarray)
        list where each array corresponds to one of the labeled regions bounding box + the border region
    '''

    # Get region props
    reg = regionprops(labeled, coordinates='rc')

    # Initialize list of images
    img_list = []

    # Get original image size
    max_col = img.shape[1]
    max_row = img.shape[0]

    for region in reg:
        if region.eccentricity < ecc_cutoff:
            (min_row, min_col, max_row, max_col) = region.bbox
            img_list.append(img[np.max([min_row-border,0]):np.min([max_row+border,max_row]),np.max([min_col-border,0]):np.min([max_col+border,max_col])])

    return img_list


