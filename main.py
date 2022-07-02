import pandas
import numpy
from matplotlib import pyplot

# 代价列表（画图用）
cost_list = []
# 导入数据
data = pandas.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
# 初始化参数θ
theta_1 = 0
theta_0 = 0
# 代价函数
def compCost(X, Y, theta1, theta0):
    iter_sample = 0
    sum_res2 = 0
    for x in X.itertuples():
        sum_res2 = sum_res2 + numpy.power(theta1*x[1] + theta0 - Y.iloc[iter_sample, 0], 2)
        iter_sample = iter_sample + 1
    return sum_res2 / (2*len(X))
# 计算每个样本的error并求和
def compSumRes_0(X, Y, theta1, theta0):
    iter_sample = 0
    sum_res_0 = 0
    for x in X.itertuples():
        sum_res_0 = sum_res_0 + theta1*x[1] + theta0 - Y.iloc[iter_sample, 0]
        iter_sample = iter_sample + 1
    return sum_res_0
def compSumRes_1(X, Y, theta1, theta0):
    iter_sample = 0
    sum_res_1 = 0
    for x in X.itertuples():
        sum_res_1 = sum_res_1 + (theta1*x[1] + theta0 - Y.iloc[iter_sample, 0])*x[1]
        iter_sample = iter_sample + 1
    return  sum_res_1
# 获取特征值和实际值
colNum = data.shape[1]
X = data.iloc[:, 0:colNum-1]
Y = data.iloc[:, colNum-1:colNum]
# 计算代价函数
cost = compCost(X, Y, theta_1, theta_0)
# Batch Gradient Descent
def batchGradientDescent(theta1, theta0, alpha, iters):
    for i in range(iters):
        sum_res_0 = compSumRes_0(X, Y, theta1, theta0)
        sum_res_1 = compSumRes_1(X, Y, theta1, theta0)
        theta0 = theta0 - (alpha * sum_res_0 / len(X))
        theta1 = theta1 - (alpha * sum_res_1 / len(X))
        cost = compCost(X, Y, theta1, theta0)
        cost_list.append(cost)
    return theta1, theta0, cost
alpha = 0.01
iters = 2000
theta_1, theta_0, cost = batchGradientDescent(theta_1, theta_0, alpha, iters)
print('theta_0:%s, theta_1:%s, cost:%s' % (theta_0, theta_1, cost))
# 出图
fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(24,8))
ax1.scatter(data.Population, data.Profit, label='Training Data')
ax1.plot(X, theta_1*X+theta_0, label='Regression')
ax1.legend(loc=2)
ax1.set_xlabel('Population')
ax1.set_ylabel('Profit')
ax2.plot(numpy.arange(iters), numpy.asarray(cost_list), 'red')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Cost')
pyplot.suptitle('Simple Linear Regression with Gradient Descent')
pyplot.show()