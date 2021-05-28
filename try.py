import os
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

with open("acc_imp12.txt","r") as f:
    datas = f.readlines()
acc = []
for data in datas:
    data = data.strip('\n')
    data = data.split('\t')
    acc_data = float(data[1])
    acc.append(acc_data)

x= range(1,201)
plt.plot(x, acc, 'b.-')

# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('accuracy rate')


plt.show()

