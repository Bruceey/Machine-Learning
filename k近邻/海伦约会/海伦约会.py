import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def classify(intX, dataSet, labels, k):
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

def fileMatrix(filename):
    """
    将datingTestSet.txt中的原始数据文件转为训练集矩阵和标签向量,
    海伦收集的样本数据datingTestSet.txt主要包含以下3种特征：
        1.每年获得的飞行常客里程数
        2.玩视频游戏所消耗时间百分比
        3.每周消费的冰淇淋公升数
    最后一行为labels,即海伦是否喜欢这个人。didntLike为不喜欢，smallDoses为有点喜欢，largeDoses为非常喜欢
    :param filename: 文件名
    :return:训练集矩阵和标签向量
    """
    with open(filename) as f:
        # 生成每行内容的列表
        data_raw = f.readlines()
        row = len(data_raw)
        # 占位
        dataSet = np.zeros((row, 3))
        labels = np.zeros(row)
        # 遍历data_raw，并赋值给dataSet和labels
        for index in range(row):
            # data_raw[index]为类似"40920\t8.326976\t0.953952\tlargeDoses\n"的形式
            data_list = data_raw[index].strip().split('\t')
            dataSet[index, :] = data_list[:3]
            # 将didntLike、smallDoses、largeDoses分别转为1、2、3
            if data_list[-1] == "didntLike":
                labels[index] = 1
            elif data_list[-1] == "smallDoses":
                labels[index] = 2
            else:
                labels[index] = 3
        return dataSet, labels


def normData(dataSet):
    """
    如果认为数据特征的权重相同，即三种特征是同等重要的，则需要归一化
    将数据集归一化, newValue = (oldValue - min) / (max - min)
    :param dataSet: 训练集
    :return:归一化的数据集
    """
    minVals = dataSet.min(0)      # 求出每列的最小值
    maxVals = dataSet.max(0)      # 求出每列的最大值
    ranges = maxVals - minVals    # 每列中最大值与最小值的差
    normSet = (dataSet - minVals) / ranges  # 利用公式归一化
    return normSet, minVals, ranges


def datingClassTest():
    """
    测试算法：验证分类器
    :return:
    """
    # 文件名
    filename = 'datingTestSet.txt'
    # 将返回的特征矩阵和分类向量分别存储到dataSet和labels中
    dataSet, labels = fileMatrix(filename)
    # 数据归一化,返回归一化后的矩阵,数据最小值,数据范围
    normSet, minVals, ranges = normData(dataSet)
    # 取所有数据的百分之十
    ratio = 0.1
    # 获得normMat的行数
    row = normSet.shape[0]
    # 百分之十的测试数据的个数
    testNum = int(row * ratio)
    # 训练数据和标签
    trainingSet = normSet[testNum:, :]
    trainingLabels = labels[testNum:]
    # 分类错误计数
    errorCount = 0
    for i in range(testNum):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classResult = classify(normSet[i, :], trainingSet, trainingLabels, 4)
        if classResult != labels[i]:
            errorCount += 1
    prob = errorCount / testNum
    print('错误率为%f%%' %(prob*100))


def classifyPerson():
    """
    建立小程序
    :return:
    """
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 打开的文件名
    filename = 'datingTestSet.txt'
    # 打开并处理数据
    dataSet, labels = fileMatrix(filename)
    # 训练集归一化
    normSet, minVals, ranges = normData(dataSet)
    # 三维特征用户输入
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 生成NumPy数组, 测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    # 测试集归一化
    normArr = (inArr - minVals) / ranges
    # 返回分类结果
    result = resultList[int(classify(normArr, normSet, labels, 3)) - 1]
    # 打印结果
    print("你可能%s这个人" % result)


if __name__ == '__main__':
    # datingClassTest()
    classifyPerson()