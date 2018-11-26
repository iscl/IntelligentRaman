import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#matplotlib inline
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import pandas as pd
import csv
import cv2


read_path = r'F:\Hem5.csv'
read_path_tri = r'F:\Tri5.csv'
csvRead = pd.read_csv(read_path, header=None, index_col=False)
csvRead_tri = pd.read_csv(read_path_tri, header=None, index_col=False)
print(len(csvRead))

x_array = np.array(csvRead[[1]])
for i in range(2, 6):
    x_array = np.append(x_array, np.array(csvRead[[i]]), axis=1)
x_array_T = np.transpose(x_array)


y_array = np.array(csvRead_tri[[1]])
for i in range(2, 6):
    y_array = np.append(y_array, np.array(csvRead_tri[[i]]), axis=1)
y_array_T = np.transpose(y_array)


X = np.append(x_array_T, y_array_T, axis=0)
ramanshift = np.array(csvRead[[0]])
print('x_array.shape:\n', x_array.shape)
print('x_array:\n', x_array)
print('x_array_T.shape:\n', X.shape)
print('x_array_T:\n', X)
plt.figure(num=1, figsize=(8, 4))
plt.plot(X[8], label="$Tri$", color="blue", linewidth=1)
plt.figure(num=2, figsize=(8, 4))
plt.plot(X[8], label="$Tri$", color="blue", linewidth=1)

fig = plt.figure(num=3, figsize=(8, 4))
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=40, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
plt.show()


