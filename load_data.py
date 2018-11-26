import os
import glob
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_data(filepath):
    data = pd.read_csv(filepath)

    y_df = data[['molecule', 'concentration']]
    X_df = data.drop(['molecule', 'concentration'], axis=1)
    spectra = X_df['spectra'].values
    spectra = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
    X_df['spectra'] = spectra.tolist()

    # freqs = pd.read_csv('datasets/freq.csv')
    # freqs = freqs['freqs'].values

    # print(np.unique(y_df['molecule'].values))

    molecule = y_df['molecule'].values

    data = molecule
    values = array(data)
    # print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print("onehot_encoded:", onehot_encoded)
    # print("shape:", np.shape(onehot_encoded))
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)

    # print("shape of spectra:", np.shape(array(spectra)))

    # concentration = y_df['concentration'].values
    #
    # plt.plot(freqs, spectra.T)
    # plt.xlabel('Freq')
    # plt.ylabel('Intensity')
    #
    # plt.hist(concentration, bins=26)
    # plt.xlabel('Concentration')
    # print('There are %s different values of concentrations.' % np.unique(concentration).size)
    # # plt.show()
    #
    # for mol in np.unique(molecule):
    #     plt.figure()
    #     plt.hist(concentration[molecule == mol], bins=20)
    #     plt.title((mol + "- %s values of concentrations.")
    #               % np.unique(concentration[molecule == mol]).size)
    #     print(np.unique(concentration[molecule == mol]))
    #
    # print('Number of samples: %s' % len(y_df))
    # y_df.groupby('molecule').count().plot(
    #     y='concentration', kind='pie', autopct='%.2f', figsize=(5, 5))
    # plt.show()

    return spectra, onehot_encoded

if __name__ == '__main__':
    load_data('datasets/train.csv.bz2')