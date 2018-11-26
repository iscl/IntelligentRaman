import os
import os.path
import numpy as np
import matplotlib.pyplot as plt


# 输入——路径，输出——拉曼频移（Ramanshift），相对强度（Intensity）
def Read_data(path):
    # 读取txt文件路径
    name_txt = os.listdir(path)
    filename = path + '/' + name_txt[0]
    # 初始化 拉曼强度数组（行列）“Intensity”
    with open(filename, 'r') as file_to_read:
        sourceInLine = file_to_read.readlines()
        flag_str = ">>>>>Begin Spectral Data<<<<<\n"
        flag_num = sourceInLine.index(flag_str)
        c = len(sourceInLine)-flag_num-1
        r = len(name_txt)
    Intensity = np.zeros((r, c))
    Ramanshift = []
    for i in range(len(name_txt)):
        Intensity_0 = []
        for line in sourceInLine[flag_num+1:len(sourceInLine)]:
            line = line.strip('\n')
            Tab_Pos = line.find('\t')
            if i == 1:
                Ramanshift.append(float(line[0:Tab_Pos]))
                pass
            Intensity_0.append(float(line[Tab_Pos+1:len(line)]))
        # 将数据从list类型转换为array类型。
        Intensity_0 = np.array(Intensity_0)
        Intensity[i] = Intensity_0
    Ramanshift = np.array(Ramanshift)
    return Ramanshift, Intensity


if __name__ == "__main__":
    print('Read_data function')
    path = "./datasets/20180803_FMYPJC/Triglyceride"
    Ramanshift, Intensity = Read_data(path)
    Intensity_F = np.mean(Intensity, axis=0)
    print(Ramanshift)
    print(Intensity_F)
    # 绘图 #
    x = Ramanshift
    y = Intensity_F
    # 控制图形的长和宽单位为英寸，
    # 调用figure创建一个绘图对象，并且使它成为当前的绘图对象。
    plt.figure(num=3, figsize=(8, 4))
    # 可以让字体变得跟好看
    # 给所绘制的曲线一个名字，此名字在图示(legend)中显示。
    # 只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
    # color : 指定曲线的颜色
    # linewidth : 指定曲线的宽度
    plt.plot(x, y, label="$background$", color="red", linewidth=2)
    # b-- 曲线的颜色和线型
    # plt.plot(x,z,"b--",label="$cos(x^2)$")
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
