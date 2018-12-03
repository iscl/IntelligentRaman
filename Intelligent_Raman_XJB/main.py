# coding=gbk
import os
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 信号处理库 #
import scipy.signal as sp
# 自建库 #
import load_data
import preprocessing_data as ppd
import rampy


if __name__ == "__main__":
    print('main function')
    path = r"F:\datasets\20181128_ABZZJC_surgery\cancer_tissue\position3"
    Ramanshift, Intensity, base_Intensity = ppd.preprocess(path, 1, 824, 932)
    # save = pd.DataFrame(Intensity)
    # save.to_csv('./assets/background.csv', index=False, header=False)


    # 绘图 #
    # 控制图形的长和宽单位为英寸，
    # 调用figure创建一个绘图对象，并且使它成为当前的绘图对象。
    plt.figure(num=1, figsize=(8, 4))
    # 可以让字体变得跟好看
    # 给所绘制的曲线一个名字，此名字在图示(legend)中显示。
    # 只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
    # color : 指定曲线的颜色
    # linewidth : 指定曲线的宽度
    # plt.plot(Ramanshift, base_Intensity, label="$Tri$", color="blue", linewidth=1)
    plt.plot(Ramanshift, Intensity, label="$Tri$", color="blue", linewidth=1)
    # 设置X轴的文字
    plt.xlabel("Raman shift/cm-1")
    # 设置Y轴的文字
    plt.ylabel("Intensity")
    # 设置图表的标题
    plt.title("Raman spectrum")
    # 设置Y轴的范围
    plt.ylim()
    # 显示图示
    plt.legend()
    # 显示出我们创建的所有绘图对象。
    plt.show()


