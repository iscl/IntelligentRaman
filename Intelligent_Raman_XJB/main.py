# coding=gbk
import os
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# �źŴ���� #
import scipy.signal as sp
# �Խ��� #
import load_data
import preprocessing_data as ppd
import rampy


if __name__ == "__main__":
    print('main function')
    path = r"F:\datasets\20181128_ABZZJC_surgery\cancer_tissue\position3"
    Ramanshift, Intensity, base_Intensity = ppd.preprocess(path, 1, 824, 932)
    # save = pd.DataFrame(Intensity)
    # save.to_csv('./assets/background.csv', index=False, header=False)


    # ��ͼ #
    # ����ͼ�εĳ��Ϳ�λΪӢ�磬
    # ����figure����һ����ͼ���󣬲���ʹ����Ϊ��ǰ�Ļ�ͼ����
    plt.figure(num=1, figsize=(8, 4))
    # �����������ø��ÿ�
    # �������Ƶ�����һ�����֣���������ͼʾ(legend)����ʾ��
    # ֻҪ���ַ���ǰ�����"$"���ţ�matplotlib�ͻ�ʹ������Ƕ��latex������Ƶ���ѧ��ʽ��
    # color : ָ�����ߵ���ɫ
    # linewidth : ָ�����ߵĿ��
    # plt.plot(Ramanshift, base_Intensity, label="$Tri$", color="blue", linewidth=1)
    plt.plot(Ramanshift, Intensity, label="$Tri$", color="blue", linewidth=1)
    # ����X�������
    plt.xlabel("Raman shift/cm-1")
    # ����Y�������
    plt.ylabel("Intensity")
    # ����ͼ��ı���
    plt.title("Raman spectrum")
    # ����Y��ķ�Χ
    plt.ylim()
    # ��ʾͼʾ
    plt.legend()
    # ��ʾ�����Ǵ��������л�ͼ����
    plt.show()


