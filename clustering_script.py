"""
Author: Joseph Morgan
Date: 19/09/2021
Title: clustering.py
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score

from DEC_AUG.metrics import acc
from Methods import feature_set, compute_clusters, cm_run

DEC_fish = np.load("Features_Results//fishfeatures_DEC.npy")
IDEC_fish = np.load("Features_Results//fishfeatures_IDEC.npy")
DEC_pigeons = np.load("Features_Results//pigeonfeatures_DEC.npy")
IDEC_pigeons = np.load("Features_Results//pigeonfeatures_IDEC.npy")

# String arrays used for plot labelling.
fish_labels = ["Catherine", "Dwayne", "Florence", "Humphrey", "Jack", "JP", "Ruby", "Selwyn", "Siobhan"]
pigeon_labels = ["Alexander", "Bertie", "Constantine", "Edward", "Friedrich", "George", "Haakon",
                 "Harald", "Henry", "James", "Nicholas", "Olav", "Oscar", "Paul", "Peter", "Wilhelm",
                 "William"]

# Configuration of data into variables as well as data formatting for use in methods and plotting.
f = h5py.File("fishes.h5", "r")
true_fish_data = f['data']
true_fish_labels = f['labels']
true_fish_names = f['names']
true_fish_data = np.asarray(true_fish_data).astype(dtype=np.single)
true_fish_data = np.reshape(true_fish_data, [-1, 128, 128, 3]) / 255
true_fish_labels = np.asarray(true_fish_labels).astype(dtype=int)
true_fish_names = np.asarray(true_fish_names).astype(dtype=str)
f.close()

f = h5py.File("pigeons.h5", "r")
true_pigeon_data = f['data']
true_pigeon_labels = f['labels']
true_pigeon_names = f['names']
true_pigeon_data = np.asarray(true_pigeon_data).astype(dtype=np.single)
true_pigeon_data = np.reshape(true_pigeon_data, [-1, 128, 128, 3]) / 255
true_pigeon_labels = np.asarray(true_pigeon_labels).astype(dtype=int)
true_pigeon_names = np.asarray(true_pigeon_names).astype(dtype=str)

f.close()

true_labels = None
legend_labels = None
clusters_c = None
images = None

"""
Automated script for generating results for each Method, Feature set and dataset.
"""

feature_sets = ["DEC", "IDEC", "HIS", "RGB"]
hog_sizes = [(2, 2), (4, 4), (8, 8)]
p_methods = ["default", "ror", "uniform"]
methods = ["KMeans", "MeanShift", "GAU", "AGG", "DBScan", "DEC", "IDEC"]
datasets = ["pigeons", "fish"]
file_results = open("plots_and_results/results_acc_DB.txt", "w")
file_results_2 = open("plots_and_results/results_nmi_DB.txt", "w")
r = 1


def run_exp(datasets, feature_sets, methods, p_methods, hog_sizes, runs):
    """
    :param datasets: The number of datasets to run the experiments on.
    :param feature_sets: The number of feature extraction/selection methods to perform on the datasets.
    :param methods: The clustering methods to run on the feature sets extracted from the datasets
    :param p_methods: The different methods to run with LBP.
    :param hog_sizes: The different `pixel per cell' sizes for running hog.
    :return:
    """
    for d in datasets:
        dataset = d
        thresh = None
        columns = 0
        eps = 0
        dn_f = []
        # Configuring experiment parameters based on which dataset is used.
        if dataset == "fish":
            true_labels = true_fish_labels
            legend_labels = fish_labels
            clusters_c = 9
            images = true_fish_data
            columns = 2
            eps = 3
            thresh = 70
        else:
            true_labels = true_pigeon_labels
            legend_labels = pigeon_labels
            clusters_c = 17
            images = true_pigeon_data
            columns = 4
            eps = 2.5
            thresh = 121
        for f in feature_sets:
            # Used to clarify which feature set is used and also formatting for a latex table.
            file_results.write(f + " & ")
            file_results_2.write(f + " & ")
            for m in methods:
                method_c = m
                # Prints out the Dataset, method and feature set currently running.
                print(d + " " + m + " " + f)
                if f == "HOG":
                    for s in hog_sizes:
                        mean_array = np.zeros([2, 1])
                        for i in range(0, runs):
                            train_data = feature_set(f, images, tr=thresh, hog_size=s)

                            data, centers, clusters = compute_clusters(method_c, train_data,
                                                                       TSNE, labels=true_labels, clusters=clusters_c,
                                                                       eps=eps)

                            mean_array[0, i] = acc(true_labels, clusters) * 100
                            mean_array[1, i] = normalized_mutual_info_score(true_labels, clusters) * 100

                        result_acc = np.mean(mean_array[0, :])
                        result_nmi = np.mean(mean_array[1, :])

                        file_results.write("{:.2f} & ".format(result_acc))
                        file_results_2.write("{:.2f} & ".format(result_nmi))

                elif f == "LBP":
                    for k in p_methods:

                        mean_array = np.zeros([2, 1])
                        for i in range(0, runs):
                            train_data = feature_set(f, images, lbp_m=k)

                            data, centers, clusters = compute_clusters(method_c, train_data,
                                                                       TSNE, labels=true_labels, clusters=clusters_c,
                                                                       eps=eps)
                            mean_array[0, i] = acc(true_labels, clusters) * 100
                            mean_array[1, i] = normalized_mutual_info_score(true_labels, clusters) * 100

                        result_acc = np.mean(mean_array[0, :])
                        result_nmi = np.mean(mean_array[1, :])

                        file_results.write("{:.2f} & ".format(result_acc))
                        file_results_2.write("{:.2f} & ".format(result_nmi))
                elif f == "DEC":
                    # Sets training data as pre extracted DEC features from running using the DNN_script.py file
                    mean_array = np.zeros([2, 1])
                    if dataset == "fish":
                        train_data = DEC_fish
                    else:
                        train_data = DEC_pigeons
                    for i in range(0, runs):
                        data, centers, clusters = compute_clusters(method_c, train_data,
                                                                   TSNE, labels=true_labels, clusters=clusters_c,
                                                                   eps=eps)

                        mean_array[0, i] = acc(true_labels, clusters) * 100
                        mean_array[1, i] = normalized_mutual_info_score(true_labels, clusters) * 100

                    result_acc = np.mean(mean_array[0, :])
                    result_nmi = np.mean(mean_array[1, :])

                    file_results.write("{:.2f} & ".format(result_acc))
                    file_results_2.write("{:.2f} & ".format(result_nmi))

                elif f == "IDEC":
                    mean_array = np.zeros([2, 1])
                    # Sets training data as pre extracted IDEC features from running using the DNN_script.py file
                    if dataset == "fish":
                        train_data = IDEC_fish
                    else:
                        train_data = IDEC_pigeons
                    for i in range(0, runs):
                        data, centers, clusters = compute_clusters(method_c, train_data,
                                                                   TSNE, labels=true_labels, clusters=clusters_c,
                                                                   eps=eps)

                        mean_array[0, i] = acc(true_labels, clusters) * 100
                        mean_array[1, i] = normalized_mutual_info_score(true_labels, clusters) * 100

                    result_acc = np.mean(mean_array[0, :])
                    result_nmi = np.mean(mean_array[1, :])

                    file_results.write("{:.2f} & ".format(result_acc))
                    file_results_2.write("{:.2f} & ".format(result_nmi))
                # This clause is to catch all other feature sets that do not require specific tuning for a run
                else:

                    mean_array = np.zeros([2, 1])
                    for i in range(0, 1):
                        train_data = feature_set(f, images, tr=thresh)

                        data, centers, clusters = compute_clusters(method_c, train_data,
                                                                   TSNE, labels=true_labels, clusters=clusters_c,
                                                                   eps=eps)

                        mean_array[0, i] = acc(true_labels, clusters) * 100
                        mean_array[1, i] = normalized_mutual_info_score(true_labels, clusters) * 100

                    result_acc = np.mean(mean_array[0, :])
                    result_nmi = np.mean(mean_array[1, :])

                    file_results.write("{:.2f} & ".format(result_acc))
                    file_results_2.write("{:.2f} & ".format(result_nmi))

            file_results.write("\\\hline")
            file_results.write("\n")
            file_results_2.write("\\\hline")
            file_results_2.write("\n")


#run_exp(datasets, feature_sets, methods, p_methods, hog_sizes, r)

file_results_2.close()
file_results.close()

# This is configuring the parameters for the single run for the confusion matrix.
dataset = "pigeon"
thresh = None
if dataset == "fish":
    true_labels = true_fish_labels
    legend_labels = fish_labels
    clusters_c = 9
    images = true_fish_data
    columns = 2
    num = [0, 1, 2, 3, 4, 5, 6, 7, 8]
else:
    true_labels = true_pigeon_labels
    legend_labels = pigeon_labels
    clusters_c = 17
    images = true_pigeon_data
    columns = 4
    num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

method_c = "DEC"
ft_set_str = "RGB9"
repeats = 1
# Font size of the texts on the plots to match the size of the text in the report.
tls = 14

data, clusters,centers,  mean_array = cm_run(ft_set_str, images, true_labels, clusters_c,eps=0,method=method_c)
C = confusion_matrix(true_labels, clusters)

M = linear_sum_assignment(-C)  # get optimal labels
ut, ua = np.unique(true_labels), np.unique(clusters)
# Relabel the assigned
na = np.zeros(len(clusters))
for i in range(len(ut)):
    na[clusters == [M[1][i]]] = i

# Creation and plotting of the confusion matrix.
Conf = confusion_matrix(true_labels, na)
plt.figure(figsize=[8, 6])
sns.heatmap(Conf, annot=True, xticklabels=legend_labels, yticklabels=legend_labels, fmt='3d')
plt.xticks(fontsize=tls)
plt.yticks(fontsize=tls)
plt.ylabel("True Label", fontsize=tls, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=tls, fontweight="bold")
plt.tight_layout()
plt.show()
# Saving the confusion matrix so to get run a mean on them. Change g to desired number
filename = "conf" + str(1) + ".npy"
np.save(filename, Conf)

plt.savefig("Fish_matrix_pre.png")
result_acc = np.mean(mean_array[0])
result_nmi = np.mean(mean_array[1])
print("{:.2f}%".format(result_acc))
print("{:.2f}%".format(result_nmi))

plt.xticks(rotation=40)
plt.figure()
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.5, 0.5]})
fig.set_size_inches(11, 8.8)
max_cl = np.max(data)
min_cl = np.min(data)

plim = max_cl
nlim = -max_cl
yoff = max_cl * 0.1

# Plot parameter tuning.
ax[0].grid(True)
ax[0].set_xlim([nlim - yoff, plim + yoff])
ax[0].set_ylim([nlim + yoff, plim - yoff])
ax[0].xaxis.set_ticklabels([])
ax[0].yaxis.set_ticklabels([])
ax[0].set_title("Clusters")

ax[1].grid(True)
ax[1].set_xlim([nlim, plim])
ax[1].set_ylim([nlim + yoff, plim - yoff])
ax[1].xaxis.set_ticklabels([])
ax[1].yaxis.set_ticklabels([])
ax[1].set_title("Original Data")

print(centers)
#Plotting of centers if they exist.
if centers:
    ax[0].scatter(centers[:, 0], centers[:, 1], c='black', s=16, alpha=0.6)
    for i in range(0, len(np.unique(clusters))):
        ax[0].text(centers[i, 0], centers[i, 1], s=np.unique(clusters)[i], fontsize=16)

# plot of data with cluster labels.
scatter_1 = ax[0].scatter(data[:, 0], data[:, 1], c=na, cmap="viridis", s=6)
# plot of the original clusters to compare to the data with cluster labels.
scatter_2 = ax[1].scatter(data[:, 0], data[:, 1], c=true_labels, cmap="viridis", s=6)

fig.subplots_adjust(hspace=10)

plt.subplots_adjust(right=0.8, wspace=0.2)
# adjusting of axis position.
box = ax[0].get_position()
box.x0 = box.x0 + 0.1
box.x1 = box.x1 + 0.1
ax[0].set_position(box)

# adjusting of axis position.
box = ax[1].get_position()
box.x0 = box.x0 + 0.1
box.x1 = box.x1 + 0.1
ax[1].set_position(box)
print(true_labels)
print(data[true_labels==6])
frame = 1223
frame_c = 4
# ax[1].scatter(data[frame,0], data[frame,1], c='black', s=12,alpha=0.6)
# ax[1].text(data[frame,0],data[frame,1], s=true_fish_names[frame], fontsize=9)
# ax[1].scatter(data[frame_c,0], data[frame_c,1], c='black', s=12,alpha=0.6)
# ax[1].text(data[frame_c,0],data[frame_c,1], s=true_pigeon_names[frame_c], fontsize=9)
# Create Legend for both axes.
ax[0].legend(handles=scatter_1.legend_elements(num=num)[0],
             labels=legend_labels, loc="upper left", bbox_to_anchor=(-0.75, 1.014), fontsize=16)



plt.show()
plt.savefig("Fish_clusters.png")

def mean_cm():

    a, b, c, d, e, f, g, h, i, j = np.load("plots_and_results/conf0.npy"), np.load("plots_and_results/conf1.npy"), np.load(
        "plots_and_results/conf2.npy"), np.load("plots_and_results/conf3.npy"), \
                                   np.load("plots_and_results/conf4.npy"), np.load("plots_and_results/conf5.npy"), np.load(
        "plots_and_results/conf6.npy"), np.load("plots_and_results/conf7.npy"), \
                                   np.load("plots_and_results/conf8.npy"), np.load("plots_and_results/conf9.npy")
    # Mean Confusion matrix for the runs.
    cm_array = np.array([a, b, c, d, e, f, g, h, i, j])
    print(cm_array.shape)
    Conf = np.mean(cm_array, axis=(0), dtype=int)
    plt.figure(figsize=[8, 6])
    sns.heatmap(Conf, annot=True, xticklabels=legend_labels, yticklabels=legend_labels, fmt='3d')
    plt.xticks(fontsize=tls)
    plt.yticks(fontsize=tls)
    plt.ylabel("True Label", fontsize=tls, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=tls, fontweight="bold")
    plt.tight_layout()

    plt.show()

    plt.savefig("pigeon_matrix.png")

mean_cm()