'''
Utility functions to save overlay information on images.
'''

import os
import numpy as np
import warnings

from skimage import io, img_as_ubyte
from skimage.color import label2rgb
from cv2 import resize, INTER_AREA

def save_overlay_image(path, image, reg_props, reg_labels):
    '''
    Generates and saves an image overlaying a green / red shade corresponding to the label passed

    Parameters
    ----------
    path: string
        Path to the save directory
    image: np.array
        Original image to use as background for the overlay
    regProps(N): list(regionProperties)
        List of regionProperties corresponding to the regions to label
    regLabels(N): list(int)
        List of binary labels (0:red, no crystal or 1:green, crystal detected) corresponding to the regionProperties
    '''

    labeled_binary = np.zeros_like(image)

    for (i, region) in enumerate(reg_props):
        np.put(labeled_binary, np.ravel_multi_index(region.coords.T, labeled_binary.shape), reg_labels[i] + 1)

    # If there are no clear droplets, then only 0 (bkgd) and 2 (crystal) are in the array.
    if 1 not in labeled_binary:
        colors = ['green']
    else:
        colors = ['red', 'green']

    overlay_label = resize(label2rgb(labeled_binary, image, colors=colors, alpha=0.2, bg_label=0),
                           None, fx=0.25, fy=0.25, interpolation=INTER_AREA)

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(path, img_as_ubyte(overlay_label))
