# -*- coding:utf-8 -*-
from sklearn import datasets
from collections import Counter  # 为了做投票
from sklearn.model_selection import train_test_split
import numpy as np

"""
datasets可以提供样本数据datasets.load_iris()导入到内存
Counter从样本选择距离最短的k个样本
"""
# 导入iris数据
iris = datasets.load_iris()  # 导入样本
X = iris.data  # 样本特征
y = iris.target  # 样本标签（预测值）
# 进一步将样本数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)
print(X_test)

# 计算欧式距离
def euc_dis(instance1, instance2):
	"""
	计算两个样本instance1和instance2之间的欧式距离
	instance1: 第一个样本， array型
	instance2: 第二个样本， array型
	"""
	# TODO
	dist = np.sqrt(np.sum(np.square(instance1 - instance2)))
	# 或**代表括号时一个向量
	dist = np.sqrt(sum((instance1 - instance2) ** 2))
	return dist


'''
给定训练数据，和测试数据testInstance，和knn算法的k值
'''


def knn_classify(X, y, testInstance, k):
	"""
	给定一个测试数据testInstance, 通过KNN算法来预测它的标签。
	X: 训练数据的特征
	y: 训练数据的标签
	testInstance: 测试数据，这里假定一个测试数据 array型
	k: 选择多少个neighbors?
	"""
	
	# TODO  返回testInstance的预测标签 = {0,1,2}
	distances = [euc_dis(x, testInstance) for x in X]
	kneighbors = np.argsort(distances)[:k]
	count = Counter(y[kneighbors])
	return count.most_common()[0][0]


# 预测结果。
predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test]
print("预测"+str(np.array(predictions)))
print('测试'+str(y_test))
print(type(y_test))
print(type(predictions))
print(np.array(predictions))
print((predictions == y_test)==True)
correct = np.count_nonzero(predictions == y_test)
print("Accuracy is: %.3f" % (correct / len(X_test)))
