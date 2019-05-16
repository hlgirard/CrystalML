'''
Methods to segment individual droplets from an array of drops in an emulsion
'''

import os
import warnings
import logging

import numpy as np

from skimage import io, exposure
from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_closing, remove_small_holes, disk
import cv2
from tqdm import tqdm


from scipy import ndimage as ndi

from src.data.utils import select_rectangle, open_grey_scale_image, crop, clear_border


def segment_skimage(img, exp_clip_limit=0.06, closing_disk_radius=4, rm_holes_area=8192, minima_minDist=100, mask_val=0.1):
    '''
    Segments droplets in an image using a watershed algorithm. Scikit-image implementation.

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
    (labeled: numpy.ndarray, num_regions: int)
        labeled: labeled array of the same shape as input image where each region is assigned a disctinct integer label.
        num_regions: number of labeled regions
    '''

    # Adaptive equalization
    img_adapteq = equalize_adapthist(img, clip_limit=exp_clip_limit)

    # Minimum threshold
    threshold = threshold_otsu(img_adapteq)

    binary = img_adapteq > threshold

    # Remove dark spots and connect bright spots
    closed = binary_closing(binary, selem=disk(closing_disk_radius))
    rm_holes_closed = remove_small_holes(closed, area_threshold=rm_holes_area, connectivity=2)

    # Calculate the distance to the dark background
    distance = ndi.distance_transform_edt(rm_holes_closed)

    # Increase contrast of the the distance image
    cont_stretch = exposure.rescale_intensity(distance, in_range='image')

    # Mask the distance image to remove interstitial points
    masked = cont_stretch.copy()
    masked[masked < mask_val] = 0

    # Find local maximas of the distance image
    local_maxi = peak_local_max(masked, indices=False, min_distance=minima_minDist)

    # Markers for watershed are the local maxima of the distance image
    markers, num_maxima = ndi.label(local_maxi)

    # Run watershed algorithm on the inverse of the distance image
    segmented = watershed(-masked, markers, mask = masked > 0)

    # Label the segments of the image
    labeled, num_regions = ndi.label(segmented)

    return (labeled, num_regions)

def segment(img, exp_clip_limit=15):
    '''
    Segments droplets in an image using a watershed algorithm. OpenCV implementation.

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of an image cropped to show only the droplets region
    exp_clip_limit: float [0-1], optional
        clip_limit parameter for adaptive equalisation

    Returns
    -------
    (labeled: numpy.ndarray, num_regions: int)
        labeled: labeled array of the same shape as input image where each region is assigned a disctinct integer label.
        num_regions: number of labeled regions
    '''

    # Adaptive Equalization
    clahe = cv2.createCLAHE(clipLimit=exp_clip_limit, tileGridSize=(8,8))
    img_adapteq = clahe.apply(img)

    # Thresholding (OTSU)
    blur = cv2.GaussianBlur(img_adapteq, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Remove small dark regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 2)
    fill_holes = ndi.morphology.binary_fill_holes(closed, structure=np.ones((3, 3))).astype('uint8')

    # Sure background area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = np.uint8(cv2.dilate(fill_holes, kernel, iterations=1))

    # Sure foreground area
    dist_transform_fg = cv2.distanceTransform(fill_holes, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform_fg, 0.25*dist_transform_fg.max(), 255, 0)
    clear_border(sure_fg)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown > 0] = 0

    # Run the watershed algorithm
    three_channels = cv2.cvtColor(fill_holes, cv2.COLOR_GRAY2BGR)
    segmented = cv2.watershed(three_channels.astype('uint8'), markers)

    return (segmented, segmented.max()-1)


def extract_indiv_droplets(img, labeled, border=25, ecc_cutoff=0.8, area_perc_cutoff=0.6):
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
    area_perc_cutoff: float, optional
        Minimum area as a percentage of the mean area

    Returns
    -------
    list(numpy.ndarray)
        list where each array corresponds to one of the labeled regions bounding box + the border region
    list(RegionProperties)
        regionProperties of the labeled regions
    '''

    # Get region props
    reg = regionprops(labeled, coordinates='rc')[1:] # First label corresponds to the background (OpenCV)

    # Initialize list of images
    img_list = []

    # Get original image size
    max_col = img.shape[1]
    max_row = img.shape[0]

    # Get area cutoff
    area_cutoff = area_perc_cutoff * np.mean([region.area for region in reg])

    reg_clean = [region for region in reg if (region.eccentricity < ecc_cutoff) and region.area > area_cutoff]

    for region in reg_clean:
        (min_row, min_col, max_row, max_col) = region.bbox
        drop_image = img[np.max([min_row-border,0]):np.min([max_row+border,max_row]),np.max([min_col-border,0]):np.min([max_col+border,max_col])]
        resized = cv2.resize(drop_image, (150,150)) * 1./255
        expanded_dim = np.expand_dims(resized, axis=2)
        img_list.append(expanded_dim)

    return img_list, reg_clean

def segment_droplets_to_file(image_filename, crop_box=None, save_overlay=False):

    if os.path.isdir(image_filename):
        img_list = [os.path.join(image_filename,f) for f in os.listdir(image_filename) if f.endswith('.JPG')]
    elif os.path.isfile(image_filename):
        img_list = [image_filename]

    # Get the crop box from the first image if not provided
    print('Getting crop box from image {}'.format(img_list[0]))
    if not crop_box:
        crop_box = select_rectangle(open_grey_scale_image(img_list[0]))

    for image_file in tqdm(img_list):
        # Open image
        image = open_grey_scale_image(image_file)

        # Obtain crop box from user if not passed as argument
        if not crop_box:
            crop_box = select_rectangle(image)

        # Crop image
        cropped = crop(image, crop_box)

        # Segment image
        (labeled, num_regions) = segment(cropped)

        # Save the overlay image if requested
        if save_overlay:
            image_overlay = label2rgb(labeled, image=cropped, bg_label=0)
            filename = image_file.split('.')[0] + '_segmented.jpg'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(filename, image_overlay)

        # Extract individual droplets
        drop_images, _ = extract_indiv_droplets(cropped, labeled)

        # Output folder has the same name as the image by default
        out_directory = image_file.split('.')[0] + '/'

        if not os.path.exists(out_directory):
            os.mkdir(out_directory)

        logging.info("Saving segmented droplets to %s", out_directory)

        # Save all the images in the output directory
        for (i, img) in enumerate(drop_images):
            name = out_directory + image_file.split('.')[0].split('/')[-1] + '_drop_' + str(i) + '.jpg' # FIXME: Make this windows compatible with os.path commands
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(name, img, check_contrast=False)
