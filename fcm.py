import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

n_samples = 3000

df = pd.read_excel('120_data.xlsx', header=None)
X = df.to_numpy()


fcm = FCM(n_clusters=3)
fcm.fit(X)

# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)

# plot result
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=fcm_labels, cmap='Set1')
plt.show()