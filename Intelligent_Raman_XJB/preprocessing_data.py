# coding=gbk
import numpy as np
import scipy.signal as sp
# �Խ��� #
import load_data
import rampy

def Normalization(data):
    # ���ݰ��бȽ�
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    numerator = data-data_min
    denominator = data_max-data_min
    Normalization_data = numerator/denominator
    return Normalization_data


def preprocess(path, mean_flag, wavenumber_Lower, wavenumber_Upper):
    print('status: pre-process start')
    # �����ļ���ȡ���� #
    Ramanshift, Intensity = load_data.Read_data(path)
    if mean_flag == 1:
        Intensity_mean = np.mean(Intensity, axis=0)
        Intensity0 = Intensity_mean
    else:
        Intensity0 = Intensity

    # ��ȡ����350~4000cm-1 #
    Lower_limit = np.max(np.where(Ramanshift < wavenumber_Lower))+1
    Upper_limit = np.min(np.where(Ramanshift > wavenumber_Upper))+1
    Ramanshift_limit = Ramanshift[Lower_limit:Upper_limit]
    Intensity0_limit = Intensity0[Lower_limit:Upper_limit]

    # SGƽ������#
    Intensity_SG = sp.savgol_filter(Intensity0_limit, 11, 2)

    # ȥ���ߴ��� #
    x = Ramanshift_limit
    y = Intensity_SG
    roi = np.array([[wavenumber_Lower, wavenumber_Upper]])
    Intensity_arpls, base_Intensity = rampy.baseline(x, y, roi, 'arPLS', lam=10 ** 6, ratio=0.001)

    # ��һ������ #
    Intensity_Normalization = Normalization(Intensity_arpls)
    base_Intensity_Normalization = Normalization(base_Intensity)
    print('status: pre-process over')
    return Ramanshift_limit, Intensity_Normalization, base_Intensity_Normalization

