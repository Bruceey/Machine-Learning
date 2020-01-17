import time
import numpy as np


def createDataSet():
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

def classify0(intX, dataSet, labels, k):
    """
        函数说明:kNN算法,分类器

        Parameters:
        	inX - 用于分类的数据(测试集)
        	dataSet - 用于训练的数据(训练集)
        	labes - 分类标签
        	k - kNN算法参数,选择距离最小的k个点
        Returns:
        	sortedClassCount[0][0] - 分类结果
        """
    # 返回dataSet的长度（即有多少行）
    dataSetSize = dataSet.shape[0]
    # 先把测试数据变为与dataSetSize相同维度的矩阵，然后相减
    diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet
    # 对相减的结果平方
    sqDiffMat = diffMat**2
    # 将矩阵每行数据分别相加求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开平方即得距离
    distances = sqDistances**0.5
    # 返回数组最小值所在的索引值
    # 如 a=np.array([1,4,2,3]), b=a.argsort(), 则b为np.array([0,2,3,1])
    sortedDistIndices = distances.argsort()
    # 对分类标签计数，如{'爱情片'：4}
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回给定的值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # reverse=True降序排序字典
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    # 返回次数最多的类别
    return sortedClassCount[0][0]


if __name__ == '__main__':
    # 计算程序运行时间
    t1 = time.perf_counter()
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 20]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)

    t2 = time.perf_counter()
    print("[Finshed in %.10fs]" % (t2 - t1))