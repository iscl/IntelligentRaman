
import numpy as np
import pandas as pds
# 信号处理库 #
import scipy.signal as sp
import scipy.io as sio
# 自建库 #
import preprocessing_data as pd
import rampy
import matplotlib.pyplot as plt


def Ttreat(Ramanshift1, Hem_I, n, Tri_path):
    Tri_I0 = np.mean(Hem_I, axis=0)
    # 截取数据350~4000cm-1, SG平滑处理 #
    Tri_I_SG = sp.savgol_filter(Tri_I0[64:1014], 11, 2)
    # 去基线处理 #
    x = Ramanshift1[64:1014]
    y1 = Tri_I_SG
    roi = np.array([[350, 4000]])
    y1_arpls, base_y1 = rampy.baseline(x, y1, roi, 'arPLS', lam=10 ** 5, ratio=0.001)
    # 归一化处理 #
    Tri_I_Nor = pd.Normalization(y1_arpls)
    Tri_I_Nor_n = np.around(Tri_I_Nor, decimals=3)
    x_ = x[:, np.newaxis]
    np.savez(Tri_path+'/' + 'Ramanspectra_  (' + str(n) + ').npz', x_=x_, Ramanspectra=Tri_I_Nor_n)
    # sio.savemat(Hem_path+'/' + 'Ramanspectra_  (' + str(n) + ').mat', {'ramanshift':x_, 'ramanspectra':Hem_I_Nor})
    # with open (Hem_path + '/all.mat', 'ab') as mt:
    #     sio.savemat(mt, {'ramanspectra'+str(n):Hem_I_Nor})
    # 下面被河蟹了
    x0 = pds.DataFrame(x, columns=['Ramanshift'])
    Tri_I_Nor = pds.DataFrame(Tri_I_Nor, columns=['Ramanspectra ' + str(n)])
    pds.merge(x0, Tri_I_Nor, how='outer', left_index=True, right_index=True). \
        to_csv(Tri_path + '/' + 'Ramanspectra_  (' + str(n) + ').csv', index=False, float_format='%.3f')
    a1 = open(Tri_path + '/all.csv')
    a = pds.read_csv(a1)
    b1 = open(Tri_path + '/' + 'Ramanspectra_  (' + str(n) + ').csv')
    b = pds.read_csv(b1)
    a.merge(b, how='outer', on='Ramanshift').to_csv(Tri_path + '/all.csv', index=False, float_format='%.3f')
    return Tri_I_Nor

def Ctreat(Ramanshift1, Cho_I, n, Cho_path):
    Cho_I0 = np.mean(Cho_I, axis=0)
    # 截取数据350~4000cm-1, SG平滑处理 #
    Cho_I_SG = sp.savgol_filter(Cho_I0[64:1014], 5, 2)
    # 去基线处理 #
    x = Ramanshift1[64:1014]
    y2 = Cho_I_SG
    roi = np.array([[350, 4000]])
    y2_arpls, base_y2 = rampy.baseline(x, y2, roi, 'arPLS', lam=10 ** 5, ratio=0.001)
    # 归一化处理 #
    Cho_I_Nor = pd.Normalization(y2_arpls)
    Cho_I_Nor_n = np.around(Cho_I_Nor, decimals=3)
    x_ = x[:, np.newaxis]
    np.savez(Cho_path + '/' + 'Ramanspectra_  (' + str(n) + ').npz', x_=x_, Ramanspectra=Cho_I_Nor_n)
    # sio.savemat(Hem_path+'/' + 'Ramanspectra_  (' + str(n) + ').mat', {'ramanshift':x_, 'ramanspectra':Hem_I_Nor})
    # with open (Hem_path + '/all.mat', 'ab') as mt:
    #     sio.savemat(mt, {'ramanspectra'+str(n):Hem_I_Nor})
    # 下面被河蟹了
    x0 = pds.DataFrame(x, columns=['Ramanshift'])
    Cho_I_Nor = pds.DataFrame(Cho_I_Nor, columns=['Ramanspectra ' + str(n)])
    pds.merge(x0, Cho_I_Nor, how='outer', left_index=True, right_index=True). \
        to_csv(Cho_path + '/' + 'Ramanspectra_  (' + str(n) + ').csv', index=False, float_format='%.3f')
    a1 = open(Cho_path + '/all.csv')
    a = pds.read_csv(a1)
    b1 = open(Cho_path + '/' + 'Ramanspectra_  (' + str(n) + ').csv')
    b = pds.read_csv(b1)
    a.merge(b, how='outer', on='Ramanshift').to_csv(Cho_path + '/all.csv', index=False, float_format='%.3f')
    return Cho_I_Nor

def Htreat(Ramanshift1, Hem_I, n, Hem_path):
    Hem_I0 = np.mean(Hem_I, axis=0)
    # 截取数据350~4000cm-1, SG平滑处理 #
    Hem_I_SG = sp.savgol_filter(Hem_I0[64:1014], 5, 2)
    # 去基线处理 #
    x = Ramanshift1[64:1014]
    y3 = Hem_I_SG
    roi = np.array([[350, 4000]])
    y3_arpls, base_y3 = rampy.baseline(x, y3, roi, 'arPLS', lam=10 ** 5, ratio=0.001)
    # 归一化处理 #
    Hem_I_Nor = pd.Normalization(y3_arpls)
    Hem_I_Nor_n = np.around(Hem_I_Nor, decimals=3)
    x_ = x[:, np.newaxis]
    np.savez(Hem_path + '/' + 'Ramanspectra_  (' + str(n) + ').npz', x_=x_, Ramanspectra=Hem_I_Nor_n)
    # sio.savemat(Hem_path+'/' + 'Ramanspectra_  (' + str(n) + ').mat', {'ramanshift':x_, 'ramanspectra':Hem_I_Nor})
    # with open (Hem_path + '/all.mat', 'ab') as mt:
    #     sio.savemat(mt, {'ramanspectra'+str(n):Hem_I_Nor})
    # 下面被河蟹了
    x0 = pds.DataFrame(x, columns=['Ramanshift'])
    Hem_I_Nor = pds.DataFrame(Hem_I_Nor, columns=['Ramanspectra ' + str(n)])
    pds.merge(x0, Hem_I_Nor, how='outer', left_index=True, right_index=True). \
        to_csv(Hem_path + '/' + 'Ramanspectra_  (' + str(n) + ').csv', index=False, float_format='%.3f')
    a1 = open(Hem_path + '/all.csv')
    a = pds.read_csv(a1)
    b1 = open(Hem_path + '/' + 'Ramanspectra_  (' + str(n) + ').csv')
    b = pds.read_csv(b1)
    a.merge(b, how='outer', on='Ramanshift').to_csv(Hem_path + '/all.csv', index=False, float_format='%.3f')
    return Hem_I_Nor

    # # 绘图 #
    # # 控制图形的长和宽单位为英寸，
    # # 调用figure创建一个绘图对象，并且使它成为当前的绘图对象。
    # plt.figure(num=1, figsize=(8, 4))
    # # 可以让字体变得跟好看
    # # 给所绘制的曲线一个名字，此名字在图示(legend)中显示。
    # # 只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
    # # color : 指定曲线的颜色
    # # linewidth : 指定曲线的宽度
    # plt.plot(x, Tri_I_Nor, label="$Tri$", color="blue", linewidth=1)
    # plt.plot(x, Cho_I_Nor + 1, label="$Cho$", color="red", linewidth=1)
    # plt.plot(x, Hem_I_Nor + 2, label="$Hem$", color="green", linewidth=1)
    #
    # plt.figure(num=2, figsize=(8, 4))
    # # 可以让字体变得跟好看
    # # 给所绘制的曲线一个名字，此名字在图示(legend)中显示。
    # # 只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
    # # color : 指定曲线的颜色
    # # linewidth : 指定曲线的宽度
    # plt.plot(Ramanshift, Tri_I0, label="$Tri$", color="blue", linewidth=1)
    # # 设置X轴的文字
    # plt.xlabel("Raman shift/cm-1")
    # # 设置Y轴的文字
    # plt.ylabel("Intensity")
    # # 设置图表的标题
    # plt.title("Raman spectrum")
    # # 设置Y轴的范围
    # plt.ylim()
    # # 显示图示
    # plt.legend()
    # # 显示出我们创建的所有绘图对象。
    # plt.show()