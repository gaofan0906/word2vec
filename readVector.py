import numpy as np
#我是用extract.py将文本转化成的300维向量并进行的保存
# 读取
x= np.load(file="./data/output/vector_x.npy")
print(x)
print(x.shape)
y= np.load(file="./data/output/vector_y.npy")
print(y)
print(y.shape)
#[1,0]negtive
#[0,1]postive
