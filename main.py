import pandas as pds
import numpy as np
import csv
import matplotlib.pyplot as plt
# 信号处理库 #
import scipy.signal as sp
import scipy.io as sio
# 自建库 #
import load_data
import preprocessing_data as pd
import rampy


if __name__ == "__main__":

    Tri_path = "./datasets/20180803_FMYPJC/Triglyceride"
    Cho_path = "./datasets/20180803_FMYPJC/Cholesterol"
    Hem_path = "./datasets/20180803_FMYPJC/Hemoglobin"
    Ramanshift10, Hem_I1 = load_data.PRead_data(Hem_path, 0)
    Ramanshift, Tri_I1 = load_data.PRead_data(Tri_path, 0)
    Ramanshift0, Cho_I1 = load_data.PRead_data(Cho_path, 0)
    x0 = Ramanshift10[64:1014]
    x_ = x0[:, np.newaxis]
    x_ = np.around(x_, decimals=3)
    x0 = pds.DataFrame(x0, columns=['Ramanshift'])
    # np.savez(Hem_path+'/all.npz', x0=x0)
    # np.savez(Hem_path+'/all.npz', x0=x0)
    # np.savez(Hem_path+'/all.npz', x0=x0)
    # sio.savemat(Hem_path + '/all.mat', {'ramanshift': x_})
    pds.DataFrame(x0, columns=['Ramanshift']).to_csv(Tri_path+'/all.csv', index=False, float_format='%.3f')
    pds.DataFrame(x0, columns=['Ramanshift']).to_csv(Cho_path+'/all.csv', index=False, float_format='%.3f')
    pds.DataFrame(x0, columns=['Ramanshift']).to_csv(Hem_path+'/all.csv', index=False, float_format='%.3f')

    print('main function')
    # for n in range(0, 2000):
    # 调用文件读取函数 #
    # Ramanshift, Tri_I = load_data.Read_data(Tri_path)
    # Ramanshift0, Cho_I = load_data.Read_data(Cho_path)
    # Ramanshift1, Hem_I = load_data.Read_data(Hem_path, n)
    load_data.TRead_data(Tri_path, x_)
    load_data.CRead_data(Cho_path, x_)
    load_data.HRead_data(Hem_path, x_)
    # pds.DataFrame(tempo0, columns=['Ramanspectra ' + str(0)]).to_csv(Hem_path + '/test.csv', index=False)




