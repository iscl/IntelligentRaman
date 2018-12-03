# coding=gbk
import numpy as np
import pandas as pd

Cholesterol_read_path = '/datasets/deal_data/Cholesterol/all.csv'
Cholesterol_csvRead = pd.read_csv(Cholesterol_read_path, header=None, index_col=False)
Hemoglobin_read_path = '/datasets/deal_data/Hemoglobin/all.csv'
Hemoglobin_csvRead = pd.read_csv(Hemoglobin_read_path, header=None, index_col=False)
Triglyceride_read_path = '/datasets/deal_data/Triglyceride/all.csv'
Triglyceride_csvRead = pd.read_csv(Triglyceride_read_path, header=None, index_col=False)
print(Triglyceride_csvRead.shape)

# 拉曼波数记录
Ramanshift = np.array(Triglyceride_csvRead[[0][0]])
Ramanshift_data = pd.DataFrame(Ramanshift)
Ramanshift_data.to_csv(r'F:\Ramanshift1.csv', header=False, index=False)

# 甘油三酯记录
row_number1 =13000 #Triglyceride_csvRead.shape[1]
Triglyceride_label = np.ones((row_number1, 1), dtype=np.int16)
str_array=",".join(map(str, np.array(Triglyceride_csvRead[[1][0]])))
Ramandata1 = np.array(str_array)
for i in range(2, row_number1+1):
    str_array = ",".join(map(str, np.array(Triglyceride_csvRead[[i][0]])))
    b = np.array(str_array)
    Ramandata1 = np.append(Ramandata1, b)
Ramandata1 = Ramandata1[:, np.newaxis]
Ramandata1 = np.append(Ramandata1, Triglyceride_label, axis=1)
print('Ramandata1.shape:\n', Ramandata1.shape)

# 血红蛋白记录
row_number2 = 13000#Hemoglobin_csvRead.shape[1]
Hemoglobin_label = 2*np.ones((row_number2, 1), dtype=np.int16)
str_array = ",".join(map(str, np.array(Hemoglobin_csvRead[[1][0]])))
Ramandata2 = np.array(str_array)
for i in range(2, row_number2+1):
    str_array = ",".join(map(str, np.array(Hemoglobin_csvRead[[i][0]])))
    b = np.array(str_array)
    Ramandata2 = np.append(Ramandata2, b)
Ramandata2 = Ramandata2[:, np.newaxis]
Ramandata2 = np.append(Ramandata2, Hemoglobin_label, axis=1)
print('Ramandata2.shape:\n', Ramandata2.shape)

# 胆固醇记录
row_number3 =13000# Cholesterol_csvRead.shape[1]
Cholesterol_label = 3*np.ones((row_number3, 1), dtype=np.int16)
str_array = ",".join(map(str, np.array(Cholesterol_csvRead[[1][0]])))
Ramandata3 = np.array(str_array)
for i in range(2, row_number3+1):
    str_array = ",".join(map(str, np.array(Cholesterol_csvRead[[i][0]])))
    b = np.array(str_array)
    Ramandata3 = np.append(Ramandata3, b)
Ramandata3 = Ramandata3[:, np.newaxis]
Ramandata3 = np.append(Ramandata3, Cholesterol_label, axis=1)
print('Ramandata3.shape:\n', Ramandata3.shape)

# 组合
Ramandata = np.vstack((Ramandata1, Ramandata2))
Ramandata = np.vstack((Ramandata, Ramandata3))
print('Ramandata.shape:\n', Ramandata.shape)

# 记录
Ramanshift_data = pd.DataFrame(Ramandata, columns=['Intensity ', 'label'])
Ramanshift_data.to_csv(r'F:\Wavenumber1.csv', index=False)
