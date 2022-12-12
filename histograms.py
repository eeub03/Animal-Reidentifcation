"""
Author: Joseph Morgan
Date: 19/09/2021
Title: histograms.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File("pigeons.h5", "r")
true_pigeon_data = f['data']
true_pigeon_data = np.asarray(true_pigeon_data).astype(dtype=np.single)
true_pigeon_data = np.reshape(true_pigeon_data, [-1, 128, 128, 3])
f.close()


f = h5py.File("fishes.h5", "r")
true_fish_data = f['data']
true_fish_data = np.asarray(true_fish_data).astype(dtype=np.single)
true_fish_data = np.reshape(true_fish_data, [-1, 128, 128, 3])
f.close()

# plt.figure(
# )
# plt.hist(true_pigeon_data)

"""
Code used was made by user 
"ptrblck" @ https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600
Modified for use with the datasets
"""
tls = 13
nb_bins = 256

count_r = np.zeros(nb_bins)
count_g = np.zeros(nb_bins)
count_b = np.zeros(nb_bins)
print(len(true_fish_data))
data = true_pigeon_data
loop = len(data)

for image in range(loop):
    x = data[image]
    hist_r = np.histogram(x[:,:,0], bins=nb_bins, range=[0, 255])
    hist_g = np.histogram(x[:,:,1], bins=nb_bins, range=[0, 255])
    hist_b = np.histogram(x[:,:,2], bins=nb_bins, range=[0, 255])
    # Counting the number of occurrences of each colour to plot.
    count_r += hist_r[0]
    count_g += hist_g[0]
    count_b += hist_b[0]

bins = hist_r[1]
fig = plt.figure()
plt.bar(bins[:-1], count_r, color='r', alpha=0.33,label="Red")
plt.bar(bins[:-1], count_g, color='g', alpha=0.33,label="Green")
plt.bar(bins[:-1], count_b, color='b', alpha=0.33,label="Blue")
plt.xlabel("Pixel Value",fontsize = tls)
plt.ylabel("Count", fontsize = tls)
plt.legend(fontsize = tls)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.show()