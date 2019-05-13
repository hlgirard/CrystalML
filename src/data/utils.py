'''
Utility functions for image manipulation.
'''

from datetime import datetime
from PIL import Image

import cv2
import matplotlib
matplotlib.use('Qt4Agg', force=True)
from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import logging


class RectangleSelection(object):
    def __init__(self, img):
        self.rectangle = None
        self.img = img
        self.done = False

        #Setup the figure
        self.fig, self.ax = plt.subplots()
        self.fm = plt.get_current_fig_manager()
        plt.ion
        plt.imshow(self.img, cmap='gray')

        self.RS = RectangleSelector(self.ax, self.onselect,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

        plt.connect('key_press_event', self.toggle_selector)
        plt.show()

    def onselect(self, e_click, e_release):
        minRow = int(min(e_click.ydata, e_release.ydata))
        minCol = int(min(e_click.xdata, e_release.xdata))
        maxRow = int(max(e_click.ydata, e_release.ydata))
        maxCol = int(max(e_click.xdata, e_release.xdata))
        self.rectangle = (minRow, minCol, maxRow, maxCol)

    def toggle_selector(self, event):
        if event.key in ['Q', 'q'] and self.RS.active:
            self.RS.set_active(False)
            self.done = True

    def close(self):
        logging.debug("Closing selection window")
        plt.close('all')

def select_rectangle(img):
    """
    Prompts the user to make a rectangular selection on the passed image

    Parameters
    ----------
    img : np.array
        image to process

    Returns
    -------
    tuple
        Rectangle coordinates following the numpy array convention (minRow, minCol, maxRow, maxCol)
    """

    print('Select the region of interest then press Q/q to confirm selection and exit.')

    selector = RectangleSelection(img)
    while not selector.done:
        pass
    crop_box = selector.rectangle
    logging.debug("Got crop_box %s", str(crop_box))
    selector.close()
    plt.pause(0.5)

    return crop_box

def get_date_taken(path):
    '''Return the date image was taken from EXIF data'''
    return datetime.strptime(Image.open(path)._getexif()[36867], '%Y:%m:%d %H:%M:%S')

def open_grey_scale_image(path):
    '''Opens an image and converts it to ubyte and greyscale'''
    image = cv2.imread(path, 0)
    if image is None:
        raise OSError("File does not exist or is not an image: {}".format(path))
    return image

def crop(img, crop_box):
    '''Returns a cropped image for crop_box = (minRow, maxRow, minCol, maxCol)'''
    (minRow, minCol, maxRow, maxCol) = crop_box
    return img[minRow:maxRow, minCol:maxCol]

def clear_border(image):
    '''Removes connected (white) items from the border of an image.'''
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for i in range(h-1): # Iterate on the lines
        if image[i, 0] == 255:
            cv2.floodFill(image, mask, (0, i), 0)
        if image[i, w-1] == 255:
            cv2.floodFill(image, mask, (w-1, i), 0)
    for i in range(w-1): # Iterate on the columns
        if image[0, i] == 255:
            cv2.floodFill(image, mask, (i, 0), 0)
        if image[h-1, i] == 255:
            cv2.floodFill(image, mask, (i, h-1), 0)
    return image