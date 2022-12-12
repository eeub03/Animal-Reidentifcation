"""
Author: Joseph Morgan
Date: 19/09/2021
Title: Methods.py
"""
import matplotlib.colors as plt_colors
import numpy as np
import pandas as pd
import scipy as scp
from skimage import color
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from tensorflow.keras.optimizers import Adam

from DEC_AUG import FcDEC, FcIDEC
from DEC_AUG.metrics import acc

"""
Functions used throughout script
"""
def compute_clusters(cluster_method, tt_data, dimension_method, labels=None, clusters=0, eps=0):
    """
    Performs the clustering method on the dataset/feature set, also perfoms dimensionality reduction.
    :param cluster_method: The Clustering method to choose.
    :param tt_data: The dataset/feature set to perform the clustering on.
    :param dimension_method: The method to reduce the dimensionality of the data.
    :param labels: (OPTIONAL)  parameter, if given, used in supervised clustering.
    :param clusters: (OPTIONAL) The number of clusters to tell the algorithm to use.
    :param eps: (OPTIONAL) If using DBScan or another distance algorithm, this is the distance between the points to cluster.
    :return data_x: The data with reduced dimensionality.
    :return pred_centers: The centers of the clustering if applicable.
    :return pred_clusters: The cluster labels predicted by the algorithm.
    """
    dm = dimension_method
    fish_state = 25
    data_x = dm(n_components=2).fit_transform(tt_data)
    # Just a message to let user know that no labels have been supplied so the run is unsupervised.
    if labels is None:
       print("No labels given")
    if cluster_method == "KMeans" or cluster_method == "MeanShift":
        if cluster_method == "KMeans":
            method = KMeans(n_clusters=clusters)
        else:
            method = MeanShift()
        CLS = method.fit(data_x, y=labels)
        pred_clusters = CLS.predict(data_x)
        pred_centers = CLS.cluster_centers_
    # If the method is a DNN then pretraining and compiling of the model needs to be done before performing the
    # clustering.
    elif cluster_method == "DEC" or cluster_method == "IDEC":
        if cluster_method == "DEC":
            CLS = FcDEC.FcDEC(dims=[data_x.shape[-1], data_x.shape[1]], n_clusters=clusters)
        else:
            CLS = FcIDEC.FcIDEC(dims=[data_x.shape[-1], data_x.shape[1]], n_clusters=clusters)
        CLS.compile()

        #CLS.pretrain(data_x, labels, optimizer=Adam(), epochs=200, batch_size=256, save_dir="DEC_AUG//results//temp")
        CLS.load_weights("DEC_AUG//results//temp//model_0.h5")
        CLS.fit(data_x, save_dir="DEC_AUG//results//temp")
        pred_centers = None
        pred_clusters = CLS.predict_labels(data_x)
    # If not a DNN or a Centered clustering method then proceed with the rest.
    else:
        if cluster_method == "DBScan":
            method = DBSCAN(eps=eps)
        elif cluster_method == "AGG":
            method = AgglomerativeClustering(n_clusters=clusters)
        else:
            method = GaussianMixture(n_components=clusters)
        pred_clusters = method.fit_predict(data_x, y=labels)
        pred_centers = None

    return data_x, pred_centers, pred_clusters


def feature_extraction(method, data):
    """
    Performs the feature extraction on the dataset for the histograms features.
    :param method: The method being used to extract the features.
    :param data: The dataset to use the feature extraction method.
    :return features: The new feature set to perform clustering on.
    """
    bin_range = 256
    features = np.zeros(shape=(len(data), bin_range, 3))

    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    # A generator is used here to save up memory in the RAM, allowing for looping over the data without causing memory
    # issues. Acts the same as using a for loop.
    def gen():
        for n in data:
            histogram_array = np.zeros(shape=[bin_range, 3])
            for channel_id, c in zip(channel_ids, colors):
                histogram, bin_edges = np.histogram(
                    n[:, :, channel_id], bins=bin_range, range=(0, 255)
                )
                histogram_array[:, channel_id] = histogram
            yield histogram_array

    for i, el in enumerate(gen()): features[i] = el
    # For the Hist_rgb set the features need to be reshaped in order to be evaluated due to the formatting of the data
    if method == "hist_rgb":
        HSV = np.asarray(features)
        features = np.reshape(HSV, [len(data), 768])
    else:
        HSV = np.asarray(plt_colors.rgb_to_hsv(features / 255))

        features = np.asarray(HSV[:, :, 0])

    return features


def get_rgb_hue_LUCY(dataset, method_name):
    return np.asarray(pd.read_csv("RGB_HUE_Features//" + dataset + method_name + ".csv", header=None))


def feature_set(method_f, images_f, tr=None, hog_size=None, lbp_m=None):
    """
    The purpose of this function is to set parameters for the feature extraction function.
    :param method_f: The method being used for the feature extraction.
    :param images_f: The images to perform the extraction on.
    :param tr: (OPTIONAL) The threshold to use if thresholding is the feature extraction method.
    :param hog_size: (OPTIONAL) The pixel per cell size to use if HOG is being used.
    :param lbp_m: (OPTIONAL) The method to be used in LBP.
    :return features: Returns the feature set
    """
    if method_f == "hist_rgb":
        features = feature_extraction(method_f, images_f)
    elif method_f == "hist_hue":
        features = feature_extraction(method_f, images_f)
    elif method_f == "HIS10" or method_f == "HIS5" or method_f == "RGB9" or method_f == "RGB4":
        if len(images_f) < 3000:
            fts = "Koi_fish"
        else:
            fts = "Pigeons"
        features = get_rgb_hue_LUCY(fts, method_f)
        print(features.shape)
    elif method_f == "HOG":
        if hog_size == (2, 2):
            size = 142884
        elif hog_size == (4, 4):
            size = 34596
        else:
            size = 8100
        features = np.zeros((len(images_f), size))

        # A generator is used here to save up memory in the RAM, allowing for looping over the data without causing memory
        # issues. Acts the same as using a for loop.
        def gen_hog():
            i = 0
            for n in images_f:
                image_output = hog(n, pixels_per_cell=hog_size, orientations=9, cells_per_block=(2, 2))
                i += 1
                yield image_output

        for f, el in enumerate(gen_hog()): features[f] = el

    else:
        images_grayscale = np.zeros((len(images_f), 128, 128))

        # A generator is used here to save up memory in the RAM, allowing for looping over the data without causing memory
        # issues. Acts the same as using a for loop.
        def gen_grayscale():
            for x in images_f:
                image_gen = color.rgb2gray(x)
                yield image_gen

        for j, el in enumerate(gen_grayscale()): images_grayscale[j] = el
        if method_f == "threshold":
            images_grayscale[images_grayscale > tr] = 1
            images_grayscale[images_grayscale <= tr] = 0
            features = images_grayscale

        elif method_f == "LBP":
            features = np.zeros((len(images_f), 128, 128))

            # A generator is used here to save up memory in the RAM, allowing for looping over the data without causing
            # memory issues. Acts the same as using a for loop.
            def gen_lbp():
                radius = 2
                for n in images_grayscale:
                    image_output = local_binary_pattern(n, radius * 8, radius, method=lbp_m)
                    yield image_output

            for f, el in enumerate(gen_lbp()): features[f] = el
        else:
            features = np.zeros((len(images_grayscale), 128, 128))

            # A generator is used here to save up memory in the RAM, allowing for looping over the data without causing memory
            # issues. Acts the same as using a for loop.
            def gen_2():
                for n in images_grayscale:
                    image_output = np.zeros(shape=n.shape)
                    scp.ndimage.sobel(n, axis=-1, output=image_output, mode='reflect', cval=0.0)

                    yield image_output

            for f, el in enumerate(gen_2()): features[f] = el

        features = np.reshape(features, [len(images_f), 16384])

    return features


def cm_run(ft, imgs, tl, cl_c, eps, method, ):
    """
    This function is to act as a single run for the experiment for use in plotting a confusion matrix.
    :param ft: Feature set to run on
    :param imgs: Data set of images to run on
    :param tl: True labels
    :param cl: Cluster labels
    :param cl_c: Stands for cluster count, means number of clusters.
    :param eps: Optional Parameter if using DBScan, is the distance between points used for the clustering.
    :param ll: Legend Labels, a list of strings used to labels the clusters.
    :return: clusters, data, centers
    """
    mean_array = np.zeros(2)

    # Font size for labeling of plots

    train_data = feature_set(ft, imgs, hog_size=(8, 8), lbp_m="default")

    data, ct, cl = compute_clusters(method, train_data,
                                    TSNE, labels=tl, clusters=cl_c, eps=eps, )
    # Stores the Accuracy and NMI of the run.
    mean_array[0] = acc(tl, cl) * 100
    mean_array[1] = normalized_mutual_info_score(tl, cl) * 100

    return data, cl, ct, mean_array
