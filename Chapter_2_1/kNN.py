# coding=utf-8
from numpy import *
import operator


def createDataSet():
    # group为特征
    group = array([
        [1.0, 1.1],
        [1.0, 1.1],
        [0, 0],
        [0, 0.1]
    ])
    # labels 为目标值
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 2-1 k邻近算法
def classify0(inX, dataSet, labels, k):
    # 计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile会将inX扩展为dataSetSize行，1列的矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 将所有列相加
    distances = sqDistances ** 0.5

    # 选择距离最小的k个点，统计这k个点对应的各类的总数
    sortedDistIndicies = distances.argsort()  # distances中的值按从小到大排序，取其排序后的值对应的原坐标作为新集合
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 排序，取总数最多的一类为目标值
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 2-2 将文本记录转换为NumPy
def file2matrix(filename):
    with open(filename, "r") as f:
        arrayOLines = f.readlines()
    numberOfLines = len(arrayOLines)  # 获取行数（样本数）
    returnMat = zeros((numberOfLines, 3))  # 生成numberOfLines行，3列的零矩阵
    classLabelVector = []  # 保存目标值
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")  # 特征值+目标值
        returnMat[index, :] = listFromLine[0:3]  # 填充零矩阵中的第index整行为特征值（3个）
        classLabelVector.append(listFromLine[-1])  # 填充该样本对应的目标值
        index += 1
    return returnMat, label2base10number(classLabelVector)


# 将labels转换为十进制整数
def label2base10number(labels):
    unique_labels = []
    labels_number = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
        labels_number.append(unique_labels.index(label) + 1)
    return labels_number


# 2-2 绘图
def draw():
    import matplotlib.pyplot as plt

    dating_data_mat, datingLabel = file2matrix("datingTestSet.txt")

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []

    for i in range(len(datingLabel)):
        if datingLabel[i] == 1:  # 第i行的label为1时
            type1_x.append(dating_data_mat[i][0])
            type1_y.append(dating_data_mat[i][1])
        if datingLabel[i] == 2:  # 第i行的label为2时
            type2_x.append(dating_data_mat[i][0])
            type2_y.append(dating_data_mat[i][1])
        if datingLabel[i] == 3:  # 第i行的label为3时
            type3_x.append(dating_data_mat[i][0])
            type3_y.append(dating_data_mat[i][1])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    type1 = ax.scatter(type1_x, type1_y, s=30, c='brown')
    type2 = ax.scatter(type2_x, type2_y, s=30, c='lime')
    type3 = ax.scatter(type3_x, type3_y, s=30, c="darkviolet")

    plt.xlabel("Frequent Flyier Miles Earned Per Year")
    plt.ylabel("Percentage of Time Spent Playing Video Games")

    ax.legend((type1, type2, type3), ("DidntLike", "SmallDoses", "LargeDoses"), loc=0)

    plt.show()


# 2-3 归一化
def autoNum(dataSet):
    minVals = dataSet.min(0)  # 每一列都是同一类特征，取各列的最小值，形成一个1x3的矩阵
    maxVals = dataSet.max(0)  # 取各列的最大值，形成一个1x3的矩阵
    ranges = maxVals - minVals  # 1x3的矩阵，每个元素对应每个特征的最大值与最小值的差值
    normDataSet = zeros(shape(dataSet))  # 构建与dataSet同行同列的零矩阵，用于存储归一化后的值
    m = normDataSet.shape[0]  # m为行数，即样本数
    normDataSet = dataSet - tile(minVals, (m, 1))  # 此时normDataSet中存放的是每个样本的每个特征值与该特征最小值的差值
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 此时normDataSet中存放的是每个样本的归一化后的所有特征值
    return normDataSet, ranges, minVals


# 2-4 针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.1  # 数据集中用作测试集的比例
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNum(datingDataMat)
    m = normMat.shape[0]  # m=样本数
    numTestVecs = int(m * hoRatio)  # 测试集数量
    errorCount = 0.0    # 统计错误个数
    # 数据集中，前 numTestVecs 作为测试集，剩下的为训练集
    for i in range(numTestVecs):
        # normMat[i, :] 取的是一整行，即一个样本的所有特征；normMat[numTestVecs:m, :] 取的是 numTestVecs至m的所有行
        classifyResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 5)
        print "分类算法得到的结果为 %d, 其真实结果为 %d" % (classifyResult, datingLabels[i])
        if (classifyResult != datingLabels[i]):
            errorCount += 1.0
    print "错误率为: %f" % (errorCount / float(numTestVecs))


# 2-5 约会网站预测函数
def classifyPerson():
    resultList = ["毫无兴趣", "一般", "非常感兴趣"]
    percentTats = float(raw_input("用来打游戏花费的时间百分比？"))
    ffMiles = float(raw_input("每年获取的飞行常客里程为？"))
    iceCreame = float(raw_input("每周吃掉的冰淇淋公升数？"))
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNum(datingDataMat)
    inArray = array([ffMiles, percentTats, iceCreame])
    classifyResult = classify0((inArray - minVals) / ranges, normMat, datingLabels, 3)
    print "你对这个人的喜欢程度可能为: ", resultList[classifyResult - 1]


if __name__ == '__main__':
    classifyPerson()
