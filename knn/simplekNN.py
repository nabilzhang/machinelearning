# encoding=utf8
from kNN import *

#最简单的kNN
group, labels = createDataSet()
print classify0([0, 0], group, labels, 3)
