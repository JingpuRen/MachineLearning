"""
Creator:JingpuRen
date:2022/10/1
case:《Machine Learning In Action》
"""

imort pandas as pd
imprt numpy as np

# 将文件中的数据转化成矩阵类型
def file2Matrix (filename):
  # filename为我们要传入的文件名称，可以是路径，需要注意的是取出转义字符的影响的方法
  fr = open(filename) # 创建文件对象
  datingList = fr.readlines() # file对象readlines()函数把文件中一行的数据返回成列表中的一个元素
  dataLines = len(datingList) # 得到文件行数，也即数据元素的个数
  datingDataMat = np.zeros((dataLines,1)) # 创建datingDataMat矩阵，用来承接转换形式之后的文件中的数据
  index = 0 # 行数
  classReturn = [] # 创建分类列表来存储每个元素所属于的类型
  for line in daingList :
    line = line.strip()
    lineList = line.split('\t')
    datingDataMat[index,:] = lineList[0:3] # 提取特征
    classReturn.append(lineList[-1]) # 提取标签
    index += 1 # 索引数或者说行数+1
   return datingDataMat,classReturn # 返回得到的特征矩阵和标签分类列表

# 数据归一化处理，去除量纲对数据的影响
def normData (dataSetMat) :
  # dataSetMat表示的是我们要进行归一化的数据，结构为矩阵类型
  minValues = dataSetMat.min(0) # 获取每一列，也就是每一个特征的最小值
  maxValues = dataSetMat.max(0) # 获取每个特征的最大值
  cnt = dataSetMat.shape[0]
  ranges = maxValuse - minValues #  获得最大值和最小值的差
  newDataMat = dataSetMat - np.tile(minValues,(cnt,1)) # 应用归一化公式，将每一个数据都减去最小值然后除以差值s
  newDataMat = newDataMat/np.tile(ranges,(cnt,1))
  return newDataMat # 返回归一化后的数据矩阵

# KNN算法分类
def classify (inX,datingDataMat,labelsList,k) :
  # inX表示的是我们将要测试或者说将要预测的数据集,datingDataMat表示的是我们用来作为基准的并且已经知道分类的结果的训练集,labelsList表示的是训练集所对应的分类标签列表,K表示前几个
  # 值得注意的是1.inX和datingDataMat都是归一化之后的矩阵，万万不要传进来原矩阵，会很大程度上受量纲的影响!!! 2.测试集或者预测集inX必须是一行一行的传进来的，不能一下子传进来整个矩阵
  m = datignDataMat.shape[0] # shape[0]是np的矩阵结构所具有的的属性，返回行数
  distance = np.tile(inX,(m,1)) - datingDataMat
  distance = distance**2
  distance = distance.sum(axis=1) # sum是矩阵结构所带有的属性，表示对一行或者一列进行相加，axis=1表示对1行进行相加,axis=0表示对1列进行相加
  distance = distance**0.5
  sortedDistance = distance.argsort() # argsort()是矩阵结构所具有的属性，表示的是返回从小到大排序后的下标，也就是说sortedDistance存储的第一个数最小的那个数的下标
  classCount = {} # 创建空字典，用来记录每个标签的发生频率
  # 总计一下前面几步的思想，先计算出距离，然后找到距离最近的前k个数据，这个时候我们统计这k个数据的分类情况并取最多的那个分类情况
  for i in range(k) :
    # 从0开始遍历sortedDistance中的前k最小的标签，并将其频数记录下来
    labelGet = labelsList[sortedDistance[i]] # 比如说当前是第0个距离最近的点，那么我们就需要把0个频率最近的点的索引取出来，然后找到这个索引对应的分类
    classCount[labelGet] = classCount.get(labelGet,0) + 1 # 记录其发生次数，如果这个标签之前遇到过，那么就加一；如果是第一次遇见，那么就返回0
  sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) # True表示降序排序
  return sortedClassCount[0][0] # 返回出现频率最高的元组的键值，也就是返回频率最高的标签

# 测试分类器性能
def datingClassTest ():
    testRate = 0.1 # 表示数据集中的10%将作为训练集
    errorCount = 0.0 # 表示分类错误的样本数
    datingDataMat,lalesList = file2Matrix(filename)
    newDataMat = normData(datingDataMat) # 将数据集进行归一化处理
    testCount = int(testRate*datingDataMat.shape[0]) # 计算测试集的数据个数，即行数
    for i in range(testCount) :
        classReturn = classify0(newDataMat[i,:],newDataMat[testCount:datingDataMat.shape[0],:],datingLabelsList[testCount:datingDataMat.shape[0]],3)
        print("came back with : %d,the real answer is : %d" % (classReturn,datingLabelsList1[i]))
        if(classReturn != datingLabelsList[i]) :
            errorCount += 1.0
    errorRate = errorCount/float(testCount) # 注意是拿错误数据的个数去除以测试集的个数，并不是除以整个数据集的个数
    return errorRate
  
  if __name__ == "__main__" :
    pass
  
