import matplotlib.pyplot as plt
import numpy as np

tolerance = 10**-6   # 精度 
C = 1                # 惩罚参数
max_iter = 300       # 最大迭代次数
alpha = np.zeros(1)  # 拉格朗日乘子
b = 0                # 阈值项
counter = 0          # 迭代计数器

# 测试数据
data = [[-7, 8],
        [-4, 9],
        [-1,1]]
data = np.array(data).T

'''
# 随机产生的测试数据
num = 30
partA = np.random.normal(1,1,(num,2))
partB = np.random.normal(5,0.8,(num,2))
output = np.array([-1]*num + [1]*num)
output.shape = (2*num, 1)
dataAB = np.concatenate((partA, partB), 0)
data = np.concatenate((dataAB, output), 1)
'''

# 保留一份原始的测试数据用于做图
input_data = data

# 核函数
def kernel(x1, x2):
    return np.dot(x1, x2)

# 当前分离超平面对于x的预测值
def func(x):
    alpha_y = np.multiply(alpha, y)
    ker = np.dot(x_data, x.T) 
    result = np.dot(alpha_y, ker)
    return result

# 剪辑新一轮的alpha
def alpha_select(x, H, L):
    if x > H :
        return H
    elif x < L :
        return L
    else:
        return x

# 选择b
def b_select(alpha_i, alpha_j, b_1, b_2):
    if alpha_i > 0 and alpha_i < C:
        return b_1
    elif alpha_j > 0 and alpha_j < C:
        return b_2
    else:
        return (b_1 + b_2) / 2.0

# 判断是否只剩下支撑向量以及是否达到最大迭代次数
while (0 in alpha) and counter < max_iter:
    # 分离特征向量和输出标签
    y = input_data[:,-1]
    x_data = input_data[:,:-1]
    num_sample = input_data.shape[0]
    sample_range = range(0, num_sample)
    alpha = np.zeros(num_sample)

    # 外循环寻找违反KKT条件的样本
    for i in sample_range :
        # 当前分离超平面对x的预测值与真实输出y的差
        E_i = func(x_data[i,:]) - y[i]
      
        # 不满足KKT条件的样本会违反下列两个条件之一
        # 在精度tolerance范围内检测违反KKT的样本
        if (y[i] * E_i < -tolerance and alpha[i] < C) or (y[i] * E_i > tolerance and alpha[i] > 0) :
            for j in sample_range :
                if j != i :  # 选择另外一个不同的样本
                    E_j = func(x_data[j, :]) - y[j]
                    
                    # 保存上一轮的alpha值用于后续tolerance精度范围内的比较
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    # 确定最优alpha的取值边界
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(C, C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] -C)
                        H = min(C, alpha[i] + alpha[j])

                    # L == H 意味着alpha[i],alpha[j]的最优解在正方形边界的角上
                    # 也即不满足支撑向量的条件：0 < alpha < C
                    if L == H :
                        continue

                    # -phi 表示样本i，j在映射空间中的欧式距离的平方，因此phi应该小于0
                    phi = 2 * kernel(x_data[i,:], x_data[j,:]) - kernel(x_data[i,:],x_data[i,:]) - kernel(x_data[j,:],x_data[j,:])
                    if phi >= 0 :
                        continue

                    # 计算沿着约束方向未经剪辑时的解
                    alpha[j] -= (y[j] * (E_i - E_j)) / phi
                    alpha[j] = alpha_select(alpha[j], H, L)

                    # 如果两个迭代结算的alpha的绝对差小于精度tolerance，则对于alpha[i]的影响过小
                    # 直接寻找下一个j
                    if abs(alpha[j] - alpha_j_old) < tolerance :
                        continue
 
                    # 根据最优解alpha[j]来更新最优的alpha[i]
                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                    # 完成alpha[i],alpha[j]的一轮优化后，重新计算阈值b
                    b_1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * kernel(x_data[i,:], x_data[i,:]) - y[j] * (alpha[j] - alpha_j_old) * kernel(x_data[i,:], x_data[j,:])
                    b_2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * kernel(x_data[i,:], x_data[j,:]) - y[j] * (alpha[j] - alpha_j_old) * kernel(x_data[j,:], x_data[j,:])
                    b = b_select(alpha[i], alpha[j], b_1, b_2)

    # 完成一轮迭代
    counter += 1
   
    # 从本次迭代的结果中选择那些alpha大于零对应的样本进行下一轮迭代
    # (alpha==0的样本不是可能的支撑变量)
    non_zero_alpha, = alpha.nonzero()
    input_data = input_data[non_zero_alpha.tolist(), :]

# while结束，根据支撑向量计算分离超平面的法向量w
w = np.dot( np.multiply(alpha, y), x_data )

# 计算分离超平面的截距项b
xx = np.linspace(min(data[:,0]), max(data[:, 0]), 300)
true_b = y[0] - np.dot(x_data[0,:], w)
yy = (np.dot(-w[0], xx) - true_b) / w[1]

# 原始样本的散点图
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])

# 分离超平面
plt.plot(xx, yy)

# 支撑向量
plt.scatter(x_data[:,0], x_data[:,1],c=['r','r'],marker='*')
plt.show()
