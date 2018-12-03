import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data_train = pd.read_csv(r'F:\iscl\IntelligentRaman\data\train.csv.bz2', index_col=False)
y_train = data_train[['molecule', 'concentration']]
y_train = y_train['molecule'].values
x_train = data_train.drop(['molecule', 'concentration'], axis=1)
spectra = x_train['spectra'].values
x_train = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
# X_train['spectra'] = spectra.tolist()
print('x_train.shape:', x_train.shape)


data_test = pd.read_csv(r'F:\iscl\IntelligentRaman\data\test.csv.bz2', index_col=False)
y_test = data_test[['molecule', 'concentration']]
y_test = y_test['molecule'].values
x_test = data_test.drop(['molecule', 'concentration'], axis=1)
spectra = x_test['spectra'].values
x_test = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
# x_test['spectra'] = spectra.tolist()
print('x_test.shape:', x_test.shape)
#
# train_accuracy = []
# test_accuracy = []
# n = []
# for n_comp in range(1, 200):
#     n.append(n_comp)
#     print(n_comp)
#     pca = PCA(n_components=n_comp)
#     pca_train = pca.fit_transform(x_train)
#     pca_test = pca.fit_transform(x_test)
#
#     lda = LDA(n_components=8)
#     lda_train = lda.fit(pca_train, y_train)
#     lda_train_accuracy = lda.score(pca_train, y_train)
#     train_accuracy.append(lda_train_accuracy)
#
#     lda_test_accuracy = lda.score(pca_test, y_test)
#     test_accuracy.append(lda_test_accuracy)


pca_n_comp = 30
pca = PCA(n_components=pca_n_comp)
pca_train = pca.fit_transform(x_train)
pca_test = pca.fit_transform(x_test)

lda = LDA(n_components=4)
lda_train = lda.fit(pca_train, y_train)
print(lda_train)
lda_train_accuracy = lda.score(pca_train, y_train)
lda_test_accuracy = lda.score(pca_test, y_test)

print('lda_train_accuracy:', lda_train_accuracy)
print('lda_test_accuracy:', lda_test_accuracy)

svm = SVC(C=1e5, gamma=0.01)
svm_train = svm.fit(pca_train, y_train)
print(svm_train)
svm_train_accuracy = svm.score(pca_test, y_test)
svm_test_accuracy = svm.score(pca_test, y_test)

print('svm_train_accuracy:', svm_train_accuracy)
print('svm_test_accuracy:', svm_test_accuracy)






# plt.plot(n, test_accuracy, linewidth=2,
#          label="LDA with shrinkage", color='r')
# plt.plot(n, train_accuracy, linewidth=2,
#          label="LDA with shrinkage", color='g')
# plt.xlabel('n_components')
# plt.ylabel('Classification accuracy')
# plt.show()