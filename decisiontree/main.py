# encoding=utf8
import trees
import treePlotter

myData, labels = trees.createDataSet()
print myData
print labels

print trees.calcShannonEnt(myData)

# myData[0][-1] = 'maybe'
# print trees.calcShannonEnt(myData)

print trees.splitDataSet(myData, 0, 1)
print trees.splitDataSet(myData, 0, 0)

print trees.chooseBestFeatureToSplit(myData)

newlabels = [x for x in labels]
treeXXXX = trees.createTree(myData, newlabels)
print treeXXXX
print "[1, 0] class is %s" % trees.classify(treeXXXX, labels, [1, 0])
print "[1, 1] class is %s" % trees.classify(treeXXXX, labels, [1, 1])

print "treePlotter.retrieveTree(1):", treePlotter.retrieveTree(1)
myTree = treePlotter.retrieveTree(1)
print "treePlotter.retrieveTree(0):", myTree

print treePlotter.getNumLeafs(myTree)
print treePlotter.getTreeDepth(myTree)

# treePlotter.createPlot(treeXXXX)
trees.storeTree(treeXXXX, "mytree.txt")

print trees.grabTree("mytree.txt")

####眼镜案例
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
print lensesTree
treePlotter.createPlot(lensesTree)


# 本例为ID3决策树算法，亏用于划分标称型数据集
