"""
Author: Joseph Morgan
Date: 19/09/2021
Title: DNN_script.py

This script was to act as a test script to use to benchmark the time and resources to use the DEC and IDEC algorithms
"""
import os
import time

import dask.array as da
import h5py
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import homogeneity_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam

from DEC_AUG import metrics, ConvIDEC, ConvDEC
# MNIST DATA LOADING
#
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x = np.concatenate((x_train, x_test))
# y = np.concatenate((y_train, y_test))
# x = x.reshape([-1, 28, 28, 1]) / 255.0

# CIFAR-10

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# print(x_train)
# x = np.concatenate((x_train, x_test))
# y = np.concatenate((y_train, y_test))
# x = x.reshape([-1, 32, 32, 3]) / 255.0
# y = y.reshape(60000)


# Fashion mnist

# (x_train,y_train), (x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
#
#
# x = np.concatenate((x_train, x_test))
# y = np.concatenate((y_train, y_test))
#
# x = x.reshape([-1, 28, 28, 1]) / 255.0
# y = y.reshape(70000)

# PIGEONS
# #
# f = h5py.File("pigeons.h5", "r")
# images = f["data"]
# labels = f["labels"]
# names = f["names"]
# print(names)
# x = np.reshape(images, [-1, 128,128,3]) / 255
# y = np.asarray(labels)
# z = np.asarray(names)
# saved_array = np.vstack([y,z])
# #
# os.chdir('C:\\Users\\Joe\\PycharmProjects\\Masters_Code\\Features_Results')
# print(saved_array.shape)
# np.save("pigeonlabels.npy",saved_array)
#
# print("finished piegons")

# fish
os.chdir('C:\\Users\\Joe\\PycharmProjects\\Masters_Code')
f = h5py.File("fishes.h5", "r")
images = f["data"]
labels = f["labels"]
names = f["names"]
print(names)
x = np.reshape(images, [-1, 128, 128, 3]) / 255
print(x.shape)
y = np.asarray(labels)
z = np.asarray(names)
os.chdir('C:\\Users\\Joe\\PycharmProjects\\Masters_Code\\Features_Results')
# saved_array = np.vstack([y,z])

# np.save("fishlabels.npy",saved_array)
print("finished fish")

algorithm = "DEC"
if algorithm == "DEC":
    CLS = ConvDEC.ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=len(np.unique(y)))
    start = time.process_time()
    CLS.compile(optimizer=Adam(), loss='kld')
    print("Time taken to compile: ", time.process_time() - start, " seconds")
    start = time.process_time()
    os.chdir("..")
    CLS.pretrain(x, y, optimizer=Adam(), epochs=2, batch_size=256, save_dir="DEC_AUG//results//temp")
    print("Time taken for pretraining: ", time.process_time() - start, " seconds")
    CLS.fit(x, y=y, save_dir="DEC_AUG//results//temp")
    print("Time taken to fit: ", time.process_time() - start, " seconds")
    # CLS.extract_features(x)
    print("Time taken to extract: ", time.process_time() - start, " seconds")
    new_labels = CLS.predict_labels(x)
    print(y.shape)
    print(new_labels.shape)
    print("Homogeneity score", homogeneity_score(y, new_labels))
    print("Accuracy Score", metrics.acc(y, new_labels))
    print("NMI Score", metrics.nmi(y, new_labels))
elif algorithm == "IDEC":
    CLS = ConvIDEC.ConvIDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=len(np.unique(y)))
    start = time.process_time()
    CLS.compile(optimizer=Adam(), loss='kld')
    print("Time taken to compile: ", time.process_time() - start, " seconds")
    start = time.process_time()
    os.chdir('DEC_AUG')
    CLS.pretrain(x, y, optimizer=Adam(), epochs=200, batch_size=256)
    print("Time taken for pretraining: ", time.process_time() - start, " seconds")
    CLS.fit(x, y=y)
    print("Time taken to fit: ", time.process_time() - start, " seconds")
    # CLS.extract_features(x)
    print("Time taken to extract: ", time.process_time() - start, " seconds")
    new_labels = CLS.predict_labels(x)
    print("Homogeneity score", homogeneity_score(y, new_labels))
    print("Accuracy Score", metrics.acc(y, new_labels))
    print("NMI Score", metrics.nmi(y, new_labels))
# os.chdir('C:\\Users\\Joe\\PycharmProjects\\Masters_Code\\Features_Results')
# np.save("pigeonfeatures_DEC.npy",CLS.extract_features(x))
