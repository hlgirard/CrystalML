from tensorflow.keras.models import model_from_json
import os
import cv2
import numpy as np
import re

# Load model from JSON
with open("models/cnn-simple-model.json", 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

# Load weights into model
model_list = sorted([model for model in os.listdir("models") if model.startswith("cnn-simple-model-") and model.endswith('.h5')], key = lambda x : int(re.search(r'\d+',x).group(0)))
print(model_list)
print("Loading model weights: {}".format(model_list[-1]))
model.load_weights("models/" + model_list[-1])

test_images_names = []
test_images = []

for test_image in sorted(os.listdir("data/test"), key = lambda x : int(re.search(r'\d+',x).group(0))) :
    img = cv2.imread("data/test/" + test_image, 0)
    img = cv2.resize(img,(150,150))
    img = np.expand_dims(img, axis=2)
    test_images.append(img*1./255)
    test_images_names.append(test_image)

test_images_array = np.asarray(test_images)

print(test_images_array.shape)

Xnew = test_images_array
ynew = model.predict_classes(Xnew)

labelDict = {1: "Crystal", 0: "Clear"}

print(ynew.flatten().tolist())

#for i in range(len(ynew)):
#    print("Image: {} - {}".format(test_images_names[i], labelDict[ynew[i][0]]))

