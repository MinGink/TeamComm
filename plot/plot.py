
# 6种方法

# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# import numpy as np
# import sys
# import glob
# df = pd.read_csv('./wandb_export_2022-01-11T22_06_07.852+08_00.csv')
# cat = ['sum', 'MIN', 'MAX']
# cols = []

# for col in df.columns[1:]:
#     if col[-3:] in cat and 'step' not in col:
#         cols.append(col)
# df = df.loc[:, cols].dropna(axis=0, how='any')
# color = ['#fca503', '#b0b0b0', '#b700ff', '#77ab3f', '#0040ff', '#ff6373']
# # txt = ['CommNet',
# # 'IC3Net',
# # 'TarMac',
# # 'MAGIC',
# # 'TieComm-w/o group',
# # 'TieComm(ours)',]
# # txt = ['Tarcomm', 'Commnet', 'IC3Net', 'Tie (Out Aproach)', 'Magic', 'Tie (No Group)']
# txt = ['TarMac', 'CommNet', 'IC3Net', 'TieComm(ours)', 'MAGIC', 'TieComm-w/o group']
# indexs = [1, 2, 0, 4, 5, 3]
# import scipy.signal
# def scipy_filter1(x,param_1=11,param_2=3):
#     new_x = scipy.signal.savgol_filter(x,param_1,param_2)
#     return new_x
# def scipy_filter2(x,param_1=7,param_2=3):
#     new_x = scipy.signal.savgol_filter(x,param_1,param_2)
#     return new_x
# def scipy_filter3(x,param_1=15,param_2=3):
#     new_x = scipy.signal.savgol_filter(x,param_1,param_2)
#     return new_x


# plt.figure(figsize=(20, 7))
# for index in indexs:
#     mean_values = []
#     max_values = []
#     min_values = []
#     values = np.squeeze(df.iloc[:, [index*3]].values)
#     scipy1 = scipy_filter1(values)
#     scipy2 = scipy_filter2(values)
#     scipy3 = scipy_filter3(values)
#     for val in zip(list(values), list(scipy1), list(scipy2), list(scipy3)):
#         val = list(val)
#         mean = sum(val) / len(val)
#         mean_values.append(mean)
        
#         variance = np.std(val)/(np.sqrt(len(val)))
#         variance *= 10
#         max_values.append(mean + variance)
#         min_values.append(mean - variance)
#     plt.plot(np.arange(len(mean_values)), values, linewidth=2.0, label=txt[index], color=color[index])
#     plt.fill_between(np.arange(len(mean_values)), np.array(min_values), np.array(max_values), color=colors.to_rgba(color[index], alpha=0.2))
# #     plt.legend()
# font1 = {'family' : 'Times New Roman',
#     'weight' : 'normal',
#     'size'   : 28,
# }

# font2 = {'family' : 'Times New Roman',
#     'weight' : 'normal',
#     'size'   : 32,
# }
# plt.legend(prop = font1)
# plt.tick_params(labelsize=26)

# plt.xlabel('Epochs', font2)
# ax=plt.gca()
# ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
# ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
# ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
# ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
# # 


# term = 'Total Reward'

# plt.ylabel(term, font2)
# plt.grid(linewidth = 2)
# plt.show()
# # plt.savefig('sample.png', dpi = 800)





#7种方法
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import sys
import glob
# df = pd.read_csv('./wandb_export_2022-01-11T22_06_07.852+08_00.csv')
df = pd.read_csv('mpev1.csv')
cat = ['sum', 'MIN', 'MAX','ean','ess']
cols = []

print(df.columns[1:])
for col in df.columns[1:]:
    if col[-3:] in cat and 'step' not in col:
        cols.append(col)
        # print(cols)
print(len(cols))
df = df.loc[:, cols].dropna(axis=0, how='any')
# print(df)
# color = ['#fca503', '#b0b0b0', '#b700ff', '#77ab3f', '#0040ff', '#ff6373','#34A514']
color = ['#fca503', '#b0b0b0', '#b700ff', '#77ab3f', '#0040ff']
# txt = ['CommNet',
# 'IC3Net',
# 'TarMac',
# 'MAGIC',
# 'TieComm-w/o group',
# 'TieComm(ours)',]
# txt = ['Tarcomm', 'Commnet', 'IC3Net', 'Tie (Out Aproach)', 'Magic', 'Tie (No Group)']
# txt = ['TarMac', 'CommNet', 'IC3Net', 'TieComm(ours)', 'MAGIC', 'TieComm-w/o group', 'GAComm']
txt = ['MAGIC', 'TarMAC', 'CommNet', 'IC3Net', 'TieComm(ours)']
# indexs = [1, 2, 0, 4, 5, 3, 6]
indexs = [1, 2, 0, 4, 3]
import scipy.signal
def scipy_filter1(x,param_1=21,param_2=3):
    new_x = scipy.signal.savgol_filter(x,param_1,param_2)
    return new_x
def scipy_filter2(x,param_1=17,param_2=3):
    new_x = scipy.signal.savgol_filter(x,param_1,param_2)
    return new_x
def scipy_filter3(x,param_1=25,param_2=3):
    new_x = scipy.signal.savgol_filter(x,param_1,param_2)
    return new_x


plt.figure(figsize=(15, 9))
for index in indexs:
    mean_values = []
    max_values = []
    min_values = []
    print(df.iloc[:, [index*2]].values)
    values = np.squeeze(df.iloc[:, [index*2]].values)
    # print(np.squeeze(df.iloc[:, [index*3]].values).shape)
    scipy1 = scipy_filter1(values)
    scipy2 = scipy_filter2(values)
    scipy3 = scipy_filter3(values)
    for val in zip(list(values), list(scipy1), list(scipy2), list(scipy3)):
        val = list(val)
        mean = sum(val) / len(val)
        mean_values.append(mean)
        
        variance = np.std(val)/(np.sqrt(len(val)))
        variance *= 10
        max_values.append(mean + variance)
        min_values.append(mean - variance)
    plt.plot(np.arange(len(mean_values)), values, linewidth=2.0, label=txt[index], color=color[index])
    plt.fill_between(np.arange(len(mean_values)), np.array(min_values), np.array(max_values), color=colors.to_rgba(color[index], alpha=0.2))
#     plt.legend()
font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
}

font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
}
plt.legend(prop = font1)
plt.tick_params(labelsize=25)

plt.xlabel('Epochs', font2)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
# 


term = 'Total Reward'

plt.ylabel(term, font2)
plt.grid(linewidth = 2)
plt.show()
# plt.savefig('sample.png', dpi = 800)