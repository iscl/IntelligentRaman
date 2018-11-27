# coding=gbk
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# #matplotlib inline
# from sklearn.datasets.samples_generator import make_blobs
# import numpy as np
# import pandas as pd
# import csv
# import cv2
#
#
# read_path = r'F:\datasets\deal_data\Cholesterol\all.csv'
# csvRead = pd.read_csv(read_path, header=None, index_col=False)
# print(len(csvRead))
#
# x_array = np.array(csvRead[[1]])
# for i in range(2, 6):
#     x_array = np.append(x_array, np.array(csvRead[[i]]), axis=1)
# x_array_T = np.transpose(x_array)
# X = x_array_T
#
# # y_array = np.array(csvRead_tri[[1]])
# # for i in range(2, 6):
# #     y_array = np.append(y_array, np.array(csvRead_tri[[i]]), axis=1)
# # y_array_T = np.transpose(y_array)
#
#
# # X = np.append(x_array_T, y_array_T, axis=0)
# ramanshift = np.array(csvRead[[0]])
# print('x_array.shape:\n', x_array.shape)
# print('x_array:\n', x_array)
# print('x_array_T.shape:\n', X.shape)
# print('x_array_T:\n', X)
# plt.figure(num=1, figsize=(8, 4))
# plt.plot(X[8], label="$Tri$", color="blue", linewidth=1)
# plt.figure(num=2, figsize=(8, 4))
# plt.plot(X[8], label="$Tri$", color="blue", linewidth=1)
#
# fig = plt.figure(num=3, figsize=(8, 4))
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=40, azim=20)
# plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
# plt.show()

import numpy as np
import pandas as pd

Cholesterol_read_path = '/datasets/deal_data/Cholesterol/all.csv'
Cholesterol_csvRead = pd.read_csv(Cholesterol_read_path, header=None, index_col=False)
Hemoglobin_read_path = '/datasets/deal_data/Hemoglobin/all.csv'
Hemoglobin_csvRead = pd.read_csv(Hemoglobin_read_path, header=None, index_col=False)
Triglyceride_read_path = '/datasets/deal_data/Triglyceride/all.csv'
Triglyceride_csvRead = pd.read_csv(Triglyceride_read_path, header=None, index_col=False)
print(Triglyceride_csvRead.shape)

# 拉曼波数记录
Ramanshift = np.array(Triglyceride_csvRead[[0][0]])
Ramanshift_data = pd.DataFrame(Ramanshift)
Ramanshift_data.to_csv(r'F:\Ramanshift.csv', header=False, index=False)

# 甘油三酯记录
row_number1 =13000 #Triglyceride_csvRead.shape[1]
Triglyceride_label = np.ones((row_number1, 1), dtype=np.int16)
str_array=",".join(map(str, np.array(Triglyceride_csvRead[[1][0]])))
Ramandata1 = np.array(str_array)
for i in range(2, row_number1+1):
    str_array = ",".join(map(str, np.array(Triglyceride_csvRead[[i][0]])))
    b = np.array(str_array)
    Ramandata1 = np.append(Ramandata1, b)
Ramandata1 = Ramandata1[:, np.newaxis]
Ramandata1 = np.append(Ramandata1, Triglyceride_label, axis=1)
print('Ramandata1.shape:\n', Ramandata1.shape)

# 血红蛋白记录
row_number2 = 13000#Hemoglobin_csvRead.shape[1]
Hemoglobin_label = 2*np.ones((row_number2, 1), dtype=np.int16)
str_array = ",".join(map(str, np.array(Hemoglobin_csvRead[[1][0]])))
Ramandata2 = np.array(str_array)
for i in range(2, row_number2+1):
    str_array = ",".join(map(str, np.array(Hemoglobin_csvRead[[i][0]])))
    b = np.array(str_array)
    Ramandata2 = np.append(Ramandata2, b)
Ramandata2 = Ramandata2[:, np.newaxis]
Ramandata2 = np.append(Ramandata2, Hemoglobin_label, axis=1)
print('Ramandata2.shape:\n', Ramandata2.shape)

# 胆固醇记录
row_number3 =13000# Cholesterol_csvRead.shape[1]
Cholesterol_label = 3*np.ones((row_number3, 1), dtype=np.int16)
str_array = ",".join(map(str, np.array(Cholesterol_csvRead[[1][0]])))
Ramandata3 = np.array(str_array)
for i in range(2, row_number3+1):
    str_array = ",".join(map(str, np.array(Cholesterol_csvRead[[i][0]])))
    b = np.array(str_array)
    Ramandata3 = np.append(Ramandata3, b)
Ramandata3 = Ramandata3[:, np.newaxis]
Ramandata3 = np.append(Ramandata3, Cholesterol_label, axis=1)
print('Ramandata3.shape:\n', Ramandata3.shape)

# 组合
Ramandata = np.vstack((Ramandata1, Ramandata2))
Ramandata = np.vstack((Ramandata, Ramandata3))
print('Ramandata.shape:\n', Ramandata.shape)

# 记录
Ramanshift_data = pd.DataFrame(Ramandata, columns=['Intensity ', 'label'])
Ramanshift_data.to_csv(r'F:\Wavenumber.csv', index=False)
