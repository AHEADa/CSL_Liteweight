import numpy as np     #导入numpy工具包
from os import listdir #使用listdir模块，用于访问本地文件
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#read dataSet
#train_dataSet, train_hwLabels = readDataSet('trainingDigits')
#knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=1)
#knn = RandomForestClassifier(random_state=0)
knn = VotingClassifier(estimators=[
    ("GNB",GaussianNB()),
    ("knn3",KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)),
    ("knn4",KNeighborsClassifier(algorithm='kd_tree', n_neighbors=4)),
    ("tree",DecisionTreeClassifier())
],voting="soft",
#weights = [0.956923076923076,0.925846153846153,0.911846153846154])
)

def knn_input(train_dataSet, train_hwLabels):
    knn.fit(train_dataSet, train_hwLabels)

def tree_input(train_dataSet, train_hwLabels):
    global dtun
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(train_dataSet, train_hwLabels)

def Gaussian_input(train_dataSet, train_hwLabels):
    global gnb
    print('Start training Bayes')
    gnb = GaussianNB().fit(train_dataSet, train_hwLabels)



'''
#read  testing dataSet
dataSet,hwLabels = readDataSet('testDigits')

res = knn.predict(dataSet)  #对测试集进行预测
error_num = np.sum(res != hwLabels) #统计分类错误的数目
num = len(dataSet)          #测试集的数目
print("Total num:",num," Wrong num:", \
      error_num,"  WrongRate:",error_num / float(num))
'''