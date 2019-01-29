import os
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage import io
from joblib import Parallel, delayed

dataDir = 'data'
labelledDir = dataDir + '/labelled'
processedDir = dataDir + '/processed'

def process_image(inDirPath, outDirPath, image):
    # Open image
    img = io.imread(inDirPath + '/' + image)
    # Convert to grayscale
    img = rgb2gray(img)
    # Resize
    img = rescale(img, 0.1, anti_aliasing = True, multichannel = False)
    # Save in processed directory
    io.imsave(outDirPath + '/' + image, img)

def process_dir(labelledDir, processedDir, direct):
    inDirPath = labelledDir + '/' + direct
    outDirPath = processedDir + '/' + direct

    if not os.path.exists(outDirPath):
        os.mkdir(outDirPath)
    
    for image in os.listdir(inDirPath):
        process_image(inDirPath, outDirPath, image)


# Process all images (one thread per category)
Parallel(n_jobs = -2, verbose = 10)(delayed(process_dir)(labelledDir, processedDir, direct) for direct in os.listdir(labelledDir))