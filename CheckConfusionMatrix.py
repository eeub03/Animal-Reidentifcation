import numpy as np
from numpy.core.fromnumeric import size 
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import matplotlib.pyplot as plt 


t = np.load('fish_true_labels.npy')
a = np.load('fish_cluster_labels.npy').astype(int)
p = np.load('plot_labels.npy')
ut,ua = np.unique(t),np.unique(a)

C = confusion_matrix(t,a)
print(np.trace(C))

M = linear_sum_assignment(-C) # get optimal labels

# Relabel the assigned
na = np.zeros(len(a))
for i in range(len(ut)):
    na[a == [M[1][i]]] = i

Conf = confusion_matrix(t,na)

plt.figure(figsize=[8,6])
sns.heatmap(Conf, annot=True, xticklabels=p, yticklabels=p, fmt = '3d')
plt.tight_layout()
plt.show()

