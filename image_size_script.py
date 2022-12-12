#!/usr/bin/env python3
from PIL import Image
import os
import sys

path = "C:\\Users\\Joe\\PycharmProjects\\Masters_Code\\animal_data\\FishImages\\Florence"
# set an initial value which no image will meet
minw = 10000000
minh = 10000000

for image in os.listdir(path):
    # get the image height & width
    image_location = os.path.join(path, image)
    im = Image.open(image_location)
    data = im.size
    # if the width is lower than the last image, we have a new "winner"
    w = data[0]
    if w < minw:
        newminw = w, image_location
        minw = w
    # if the height is lower than the last image, we have a new "winner"
    h = data[1]
    if h < minh:
        newminh = h, image_location
        minh = h
# finally, print the values and corresponding files
print("minwidth", newminw)
print("minheight", newminh)