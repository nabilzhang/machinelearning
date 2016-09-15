# encoding=utf8
import kNN

testVector = kNN.img2vector('digits/testDigits/0_13.txt')
print testVector
print testVector[0, 0:31]
print testVector[0, 32:63]

kNN.handWritingClassTest()