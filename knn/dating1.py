# encoding=utf8
import matplotlib.pyplot as plt
from numpy import *

import kNN

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print datingDataMat
print datingLabels[0:20]

##图例
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

# 归一化
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

print normMat

kNN.datingClassTest()

kNN.classifyPerson()
