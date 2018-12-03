#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
# from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 数据录入_训练集#
data_train = pd.read_csv(r'F:\iscl\IntelligentRaman\data\train.csv.bz2', index_col=False)
y_train = data_train[['molecule', 'concentration']]
y_train = y_train['molecule'].values
x_train = data_train.drop(['molecule', 'concentration'], axis=1)
spectra = x_train['spectra'].values
x_train = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
print('x_train.shape:', x_train.shape)


# 数据录入_测试集#
data_test = pd.read_csv(r'F:\iscl\IntelligentRaman\data\test.csv.bz2', index_col=False)
y_test = data_test[['molecule', 'concentration']]
y_test = y_test['molecule'].values
x_test = data_test.drop(['molecule', 'concentration'], axis=1)
spectra = x_test['spectra'].values
x_test = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
print('x_test.shape:', x_test.shape)

# PCA降维#
pca_n_comp = 30
pca = PCA(n_components=pca_n_comp)
pca_train = pca.fit_transform(x_train)
pca_test = pca.fit_transform(x_test)


# LDA分类#
lda = LDA(n_components=4)
lda_train = lda.fit(pca_train, y_train)
print(lda_train)
lda_train_accuracy = lda.score(pca_train, y_train)
lda_test_accuracy = lda.score(pca_test, y_test)
print('lda_train_accuracy:', lda_train_accuracy)
print('lda_test_accuracy:', lda_test_accuracy)


# SVM分类#
svm = SVC(C=1e5, gamma=0.01, probability=True)
svm_train = svm.fit(pca_train, y_train)
print(svm_train)
svm_train_accuracy = svm.score(pca_test, y_test)
svm_test_accuracy = svm.score(pca_test, y_test)
print('svm_train_accuracy:', svm_train_accuracy)
print('svm_test_accuracy:', svm_test_accuracy)
svm_pred = svm.predict(pca_test)
svm_prpbas = svm.predict_proba(pca_test)
print(svm_prpbas.shape)
print(y_test.shape)


# 评价部分_混淆矩阵#
cm = confusion_matrix(y_test, svm_pred, labels=None, sample_weight=None)
print(cm)


# 评价部分_混淆矩阵#
precision = precision_score(y_test, svm_pred, average=None)
recall = recall_score(y_test, svm_pred, average=None)
F1 = f1_score(y_test, svm_pred, average=None)
fpr, tpr, thresholds = roc_curve(y_test, svm_prpbas[:, 0], pos_label='A')
tnr = 1-tpr

print(tnr.shape)
print('precision_score:', precision, '\n', 'recall_score:', recall, '\n',
      'recall_score:', F1, '\n', 'Sensitivity:', fpr, '\n', 'Specificity:', tnr)


# 绘图
plt.matshow(cm, cmap=plt.cm.Reds)
plt.colorbar()
for x in range(len(cm)):
    for y in range(len(cm)):
        plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
plt.show()
