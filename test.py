# import tensorflow as tf
# import numpy as np
#
# # # 定义一个矩阵a，表示需要被卷积的矩阵。
# # # a = np.array(np.arange(1, 1 + 20).reshape([1, 10, 2]), dtype=np.float32)
# # #
# # # # 卷积核，此处卷积核的数目为1
# # # kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([2, 2, 1])
# # #
# # # print(np.shape(kernel))
# # #
# # # # 进行conv1d卷积
# # # conv1d = tf.nn.conv1d(a, kernel, stride=1, padding='VALID')
# # #
# # # with tf.Session() as sess:
# # #     # 初始化
# # #     tf.global_variables_initializer().run()
# # #     # 输出卷积值
# # #     print(sess.run(conv1d))
#
# # datafile1 = "./datasets/label.npy"
# # X = np.load(datafile1)
#
# # X = [0, 0, 1, 0, 0, 0]
# # print(np.argmax(X))
# #
# # print(X)
#
# labels = [1, 3, 4, 8, 7, 5, 2, 9, 0, 8, 7]
# one_hot_index = np.arange(len(labels)) * 10 + labels
#
# print('one_hot_index:{}'.format(one_hot_index))
#
# one_hot = np.zeros((len(labels), 10))
# one_hot.flat[one_hot_index] = 1
#
#
#
# print(one_hot.flat[1])
#
# print('one_hot:{}'.format(one_hot))

# from numpy import array
# from numpy import argmax
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# # define example
# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# values = array(data)
# print(values)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# # invert first example
# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# print(inverted)

import numpy as np
import pandas as pd

# data = pd.read_csv('datasets/test.csv.bz2')
#
# y_df = data[['molecule', 'concentration']]
# X_df = data.drop(['molecule', 'concentration'], axis=1)
# spectra = X_df['spectra'].values
# spectra = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
# X_df['spectra'] = spectra.tolist()
#
# # freqs = pd.read_csv('datasets/freq.csv')
# # freqs = freqs['freqs'].values
#
# print(np.shape(spectra))
#
# molecule = y_df['molecule'].values
#
# print(np.shape(molecule))

x_vals = np.random.normal(1, 0.1, 100)

y_vals = np.repeat(10., 100)

print(x_vals, y_vals)


