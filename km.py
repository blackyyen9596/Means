from sklearn import cluster, datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import pylab

df = pd.read_excel('120_data.xlsx', header=None)
X = df.to_numpy()

# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 3).fit(X)

# 印出分群結果
cluster_labels = kmeans_fit.labels_
# print("分群結果：")
# print(cluster_labels)
# print("---")

# plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap='Set1')
plt.show()