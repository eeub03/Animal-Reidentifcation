"""
Code provided by rosdeepy  at: https://www.pyimagedata.com/how-to-create-custom-image-dataset/
Edited by Joseph Morgan

Author: Joseph Morgan
Date: 19/09/2021
Title: images_to_dataset.py

This file is intended to turn the images into a numpy dataset with the pixel values in order to perform clustering
on them.
"""
import os
from glob import glob

import cv2
import dask.array as da
import dask.dataframe as dd
import h5py
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class CreateDataset():
    # Costructer method
    def __init__(self, class_dict, path):
        self.array = da.array([])
        self.Image_dict = class_dict
        self.labels = da.array([])
        self.path = path
        self.image_names = []

    def create_dataset(self):
        """
        Resizes images to
        :return:
        """
        for name in os.listdir(self.path):  # get image folders from directory
            print(self.path, name)  # print folder name

            for image_name in tqdm(glob(f'{self.path}/{name}/*.jpg')):  # get all image names with path from in folder
                # Read image and resize it
                image = cv2.imread(image_name)
                image = cv2.resize(image, [128, 128])

                image = da.asarray(image)

                self.array = da.append(self.array, image)
                self.labels = da.append(self.labels, self.Image_dict[name])
                string_name = f'{image_name}'

                string_image_name = string_name.split(f'{name}\\', 2)[1]

                self.image_names = da.append(self.image_names, string_image_name)

    def create_h5df(self):
        """
        Stores the dataset made as a h5 file

        """
        h5 = h5py.File("pigeons.h5", 'w')
        array = self.array
        labels = self.labels
        image_names = self.image_names
        # array = np.asarray(self.array)

        # labels = np.asarray(self.labels)

        h5['data'] = array
        h5['labels'] = labels
        dt = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset('names', dtype=dt, data=image_names)

        # img_g = h5.create_group("image_names")
        # data_g = h5.create_group("data")
        # labels_g = h5.create_group("labels")
        #
        # data_g.create_dataset("image_names", data=array)
        # labels_g.create_dataset("image_names", data=labels)
        # img_g.create_dataset("image_names",data=image_names)

        h5.close()
