from datetime import datetime

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt
from PIL import Image

class RectangleSelection(object):
    def __init__(self, img):
        self.rectangle = None
        self.img = img
        self.done = False

        #Setup the figure
        self.fig, self.ax = plt.subplots()
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

    plt.close('all')

    return selector.rectangle

def get_date_taken(path):
    '''Return the date image was taken from EXIF data'''
    return datetime.strptime(Image.open(path)._getexif()[36867],  '%Y:%m:%d %H:%M:%S')