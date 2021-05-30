from iris_data import *
from iris_plot import *
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt


class decisionnode:
    def __init__(self, dimension=None, threshold=None,results=None, NH=None,leftnode=None, rightnode=None, maxLabel=None):
        self.dimension = dimension 
        self.threshold = threshold 
        self.results = results 
        self.NH = NH
        self.leftnode = leftnode 
        self.rightnode = rightnode 
        self.maxLabel = maxLabel

# 计算信息熵
# 输入:labels(ndarray)
# 输出该属性的信息熵
def compute_entropy(labels):
    if labels.size > 1:
        category = list(set(labels))
    else:
        category = [labels.item()]
    entropy = 0
    for label in category:
        p = len([label_ for label_ in labels if label_ == label]) / len(labels)
        entropy += -p * math.log(p, 2)
    return entropy

# 计算各属性最大增益
# data(原始数据) label(对应数据的标签) 
# 返回最大信息增益对应值，以及选取的二分类值
def maxGainEntropy(data, labels, dimension):
    entropyX = compute_entropy(labels)
    attribution = data[:, dimension]
    attribution = list(set(attribution))
    attribution = sorted(attribution)
    gain = 0
    value = 0
    for i in range(len(attribution) - 1):
        value_temp = (attribution[i] + attribution[i + 1]) / 2 #取每两个的中值进行计算
        small_index = [j for j in range(
            len(data[:, dimension])) if data[j, dimension] <= value_temp]
        big_index = [j for j in range(
            len(data[:, dimension])) if data[j, dimension] > value_temp]
        small = labels[small_index]
        big = labels[big_index]
        # 计算信息增益
        gain_temp = entropyX - ((len(small) / len(labels)) * compute_entropy(small) + (len(big) / len(labels)) * compute_entropy(big))
        # 获取最大信息增益
        if gain < gain_temp:
            gain = gain_temp
            value = value_temp
    return gain, value

# 根据计算不同属性的信息增益，获取最大信息增益的属性选择类别
# 返回信息增益值，二分类数值以及对应属性
def maxAttribute(data, labels):
    Length = np.arange(len(data[0]))
    gainMax = 0
    valueMax = 0
    dimensionMax = 0
    for dimension in Length:
        gain, value = maxGainEntropy(data, labels, dimension)
        if gainMax < gain:
            gainMax = gain
            valueMax = value
            dimensionMax = dimension
    return gainMax, valueMax, dimensionMax

# 根据dimension属性中的value 对data与label 进行而分类
def devideGroup(data, labels, value, dimension):
    small_index = [j for j in range(
        len(data[:, dimension])) if data[j, dimension] <= value]
    big_index = [j for j in range(
        len(data[:, dimension])) if data[j, dimension] > value]
    dataSmall = data[small_index]
    dataBig = data[big_index]
    labelsSmall = labels[small_index]
    labelsBig = labels[big_index]
    return dataSmall, labelsSmall, dataBig, labelsBig


# 计算熵与样本数量的乘积
def product(labels):
    entropy = compute_entropy(labels)
    return entropy * len(labels)

# 获取出现次数最多的标签
def getMaxLabel(labels):
    label = Counter(labels)
    label = label.most_common(1)
    return label[0][0]

# 递归的方式构建决策树
def buildDescisionTree(data, labels):
    if labels.size > 1:
        gain_max, value_max, dimension_max = maxAttribute(data, labels)
        if (gain_max > 0) :
            dataSmall, labelsSmall,dataBig, labelsBig = devideGroup(data, labels, value_max, dimension_max)
            left_branch = buildDescisionTree(dataSmall, labelsSmall)
            right_branch = buildDescisionTree(dataBig, labelsBig)
            NH=product(labels)
            maxLabel = getMaxLabel(labels)
            return decisionnode(dimension=dimension_max, threshold=round(value_max,2), NH=NH,leftnode=left_branch, rightnode=right_branch, maxLabel=maxLabel)
        else:
            NH=product(labels)
            maxLabel = getMaxLabel(labels)
            return decisionnode(results=labels[0], NH=NH, maxLabel=maxLabel)
    else:
        NH=product(labels)
        maxLabel = getMaxLabel(labels)
        return decisionnode(results=labels.item(), NH=NH, maxLabel=maxLabel)

def printTree(tree, indent='-', dict_tree={}, direct='L'):
    if tree.results != None:
        print(tree.results)
        dict_tree = {direct: str(tree.results)}
    else:
        print("属性" + str(tree.dimension) + ":" + str(tree.threshold) + "? ")
        print(indent + "L->",)
        l = printTree(tree.leftnode, indent=indent + "-", direct='L')
        l2 = l.copy()
        print(indent + "R->",)
        r = printTree(tree.rightnode, indent=indent + "-", direct='R')
        r2 = r.copy()
        l2.update(r2)
        stri = str(tree.dimension) + ":" + str(tree.threshold) + "?"
        if indent != '-':
            dict_tree = {direct: {stri: l2}}
        else:
            dict_tree = {stri: l2}
    return dict_tree

def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.dimension]
        branch = None
        if v > tree.threshold:
            branch = tree.rightnode
        else:
            branch = tree.leftnode
        return classify(observation, branch)

def pruning(tree, alpha=0.1):
    if tree.leftnode.results == None:
        pruning(tree.leftnode, alpha)
    if tree.rightnode.results == None:
        pruning(tree.rightnode, alpha)
    if tree.leftnode.results != None and tree.rightnode.results != None:
        before_pruning = tree.leftnode.NH + tree.rightnode.NH + 2 * alpha
        after_pruning = tree.NH + alpha
        print('before_pruning={},after_pruning={}'.format(
            before_pruning, after_pruning))
        if after_pruning <= before_pruning:
            print('pruning--{}:{}?'.format(tree.dimension, tree.threshold))
            tree.leftnode, tree.rightnode = None, None
            tree.results = tree.maxLabel


if __name__ == '__main__':
    filename='iris.data'
    label_dict={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
    iris=read_iris(filename)
    data=np.array([d[:4] for d in iris])
    labels=np.array([label_dict[d[4]] for d in iris])
    array = np.random.permutation(data.shape[0])
    shuffled_data = data[array,:]
    shuffled_labels = labels[array]
    train_data = shuffled_data[:100, :]
    train_labels = shuffled_labels[:100]
    test_data = shuffled_data[100:150, :]
    test_labels = shuffled_labels[100:150]
    tree = buildDescisionTree(train_data,train_labels)
    printedTree = printTree(tree=tree)
    true_num = 0
    for i in range(len(test_labels)):
        prediction = classify(test_data[i],tree)
        if prediction == test_labels[i]:
            true_num += 1
    print("ID3Tree true_num:{}".format(true_num))
    print('accuracy={}'.format(true_num/len(test_labels)))
    createPlot(printedTree, 1)
    plt.show()