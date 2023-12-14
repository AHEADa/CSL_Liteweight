import cv2
import mediapipe as mp
import time
import math
import numpy as np
from random import shuffle
import reconize_knn
from sklearn import model_selection
from tqdm import tqdm

class Hand():

    __points = []

    def __init__(self, points, mpDraw, img, mpHands) -> None:
        self.__points  = points
        self.__mpDraw  = mpDraw
        self.__img     = img
        self.__mpHands = mpHands

    # 算三点夹角(弧度制)，point_B是顶点
    def __calculateAngle(self, point_A, point_B, point_C) -> float:

        a_Power = (point_B.x - point_C.x)**2 + (point_B.y - point_C.y)**2 + (point_B.z - point_C.z)**2
        b_Power = (point_A.x - point_C.x)**2 + (point_A.y - point_C.y)**2 + (point_A.z - point_C.z)**2
        c_Power = (point_A.x - point_B.x)**2 + (point_A.y - point_B.y)**2 + (point_A.z - point_B.z)**2
 
        a = math.sqrt(a_Power)
        c = math.sqrt(c_Power)

        cosB = (b_Power - a_Power - c_Power) / (-2 * a * c)

        return math.acos(cosB) # 弧度制

    def getAngles(self) -> list:
        __allAngles = []
        i = 0
        for i in range(5):
            # i是手指, x从指根开始
            pointPlaces = list(map(lambda x: x + 4 * i, [1, 2, 3, 4]))
            pointPlaces.insert(0, 0)
            for i in range(3):
                a = self.__points.landmark[pointPlaces[i]]
                b = self.__points.landmark[pointPlaces[i+1]]
                c = self.__points.landmark[pointPlaces[i+2]]
                __allAngles.append(self.__calculateAngle(a, b, c))

        return __allAngles

    # A为点0, B为点5
    def _calculateSlope(self, point_A, point_B) -> float:
        #print((point_B.y - point_A.y)/(point_B.x - point_A.x)/math.pi * 180)
        return (point_B.y - point_A.y)/(point_B.x - point_A.x)/math.pi * 180

    def getDirection(self) -> int:
        point_A = self.__points.landmark[0]
        point_B = self.__points.landmark[5]
        if abs(self._calculateSlope(point_A, point_B)) <= 130:
            return 0 # 水平
        else:
            return 1 # 向上

    def drawPoints(self) -> None:
        self.__mpDraw.draw_landmarks(self.__img, self.__points, self.__mpHands.HAND_CONNECTIONS)


class Dataset():

    labelList  = []

    trainSet   = []
    testSet    = []

    trainLabel = []
    testLabel  = []

    def __init__(self, labelList: list) -> None:
        self.labelList = labelList

    def readData(self, label: str) -> list:
        data = []
        r = open("{}.txt".format(label), 'r')
        for line in r.readlines():
            lineContent = []
            # 去除空白
            line = line.strip()[0:-1]
            for i in line.split(','):
                lineContent.append(float(i))
            #print(lineContent)
            data.append(lineContent)

        return data

    # ratio: 训练集/总数
    def divideDat(self, datList: list, ratio: float) -> list:
        shuffle(datList)
        num = int(len(datList) * ratio)
        # 训练集, 测试集
        return datList[0:num], datList[num:]

    # ratio: 训练集/总数
    def makeDataset(self, ratio: float) -> None:
        # 对于每一个标签
        for i in self.labelList:
            # 读取数据集
            data = self.readData(i)
            # 切分数据集
            train, test = self.divideDat(data, ratio)
            # 整理数据
            self.trainSet.extend(train)
            self.testSet.extend(test)
            # 生成标签
            self.trainLabel.extend([i]*len(train))
            self.testLabel.extend([i]*len(test))


class Timer():
    
    def __init__(self) -> None:
        self.startTime = 0
        self.endTime   = 0

    def start(self) -> None:
        self.startTime = time.perf_counter()

    def currentTime(self) -> float:
        self.endTime = time.perf_counter()
        return self.endTime - self.startTime

def test() -> float:

    rightall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wrongall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    right = 0
    wrong = 0
    for c in tqdm(range(100)):
        a = 0
        a = Dataset(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])
        a.makeDataset(0.5)
        #reconize_knn.knn_input(np.asarray(a.trainSet), np.asarray(a.trainLabel))
        #reconize_knn.tree_input(np.asarray(a.trainSet), np.asarray(a.trainLabel))
        reconize_knn.knn_input(np.asarray(a.trainSet), np.asarray(a.trainLabel))
        count = 0
        for i in a.testSet:
            #res = reconize_knn.knn.predict(np.asarray([i]))
        #rate = reconize_knn.knn.predict_proba(np.asarray([i]))
        #res = reconize_knn.dt.predict(np.asarray([i]))
            res = reconize_knn.knn.predict(np.asarray([i]))
        #rate = reconize_knn.dt.predict_proba(np.asarray([i]))
        #rate = reconize_knn.gnb.predict_proba(np.asarray([i]))
            if a.testLabel[count] == res:
                right += 1
                riall = rightall[int(a.testLabel[count])-1]
                rightall[int(a.testLabel[count])-1] = riall+1
            else:
                wrong += 1
                woall = wrongall[int(a.testLabel[count])-1]
                wrongall[int(a.testLabel[count])-1] = woall+1
            count += 1
        #print(res)
        #print(rate)



    print("rate:")
    print(right/(right+wrong))
    print(rightall)
    print(wrongall)
    return right/(right+wrong)


def crossTest(x, y):
    result = []
    for i in range(100):
        x_train, x_test, y_train, y_test = \
                    model_selection.train_test_split(x, y, test_size = 0.2)
        reconize_knn.knn_input(x_train, y_train)
        #reconize_knn.tree_input(x_train, y_train)
        #reconize_knn.Gaussian_input(x_train, y_train)

        result.append(np.mean(y_test == reconize_knn.knn.predict(x_test)))
        #result.append(np.mean(y_test == reconize_knn.dt.predict(x_test)))
        #result.append(np.mean(y_test == reconize_knn.gnb.predict(x_test)))

    print("svm classifier accuacy:")
    print(np.mean(result))


def seperateFinger(allAngles: list) -> list:

    SeperateFinger = [allAngles[0:5], allAngles[5:10], allAngles[10:15]]
    return SeperateFinger


def isGuesture(probaList: list) -> bool:
    sortedList = sorted(probaList, reverse = True)
    if sortedList[0] >= 3 * sortedList[1]:
        return True
    else:
        return False

# GuestureLeft: list [手势编号str, 方向str: 0水平 1向上]
def translate(GuestureLeft: list, GuestureRight: list) -> str:

    print(GuestureLeft)
    print(GuestureRight)

    # 右手
    rightList_0  = ['01', '03', '04', '06', '07', '08', '09', '10', '11', '12', '13']
    rightLabel_0 = ['j' , 'di', 'h' , 'ni', 'k' , 'bi', 'pi', 'mi', 'ti', 'x' , 'z' ]

    rightList_1  = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    rightLabel_1 = ['g' , 'd' , 's' , 'f' , 'l' , 'q' , 'b' , 'p' , 'm' , 't' , 'r' , 'c' ]

    # 左手
    leftList_0  = ['01', '02', '03'  , '05' , '06', '07', '08', '10' , '11', '12', '13' ]
    leftLabel_0 = ['an', 'e' , 'uang', 'uan', 'un', 've', 'un', 'uan', 'ai', 'ei', 'ong']

    leftList_1  = ['01' , '02', '03', '04' , '05', '06',  '07', '08', '09', '10', '11', '12', '13']
    leftLabel_1 = ['ang', 'er', 'ua', 'uei', 'u' , 'uai', 'ie', 'en', 'v' , 'o' , 'a' , 'ao', 'ou']

    # 本该右手在先，不知道为什么反了
    # 画面镜像过了，左右判定反了
    try:
        # 右手
        if GuestureRight[1] == 0:
            code = rightLabel_0[rightList_0.index(GuestureRight[0])]
        else:
            code = rightLabel_1[rightList_1.index(GuestureRight[0])]

        # 左手
        if GuestureLeft[1] == 0:
            code = code + leftLabel_0[leftList_0.index(GuestureLeft[0])]
        else:
            code = code + leftLabel_1[leftList_1.index(GuestureLeft[0])]

    except:
        print("err")
        code = ""

    return code.replace("xen", "xing")


def translateSingle(GuestureHand: list, handness) -> str:
    
    # 左手
    leftList_0  = ['01', '02', '03'  , '05' , '06' , '07' , '08' , '10'  , '11', '12', '13'  ]
    leftLabel_0 = ['an', 'e' , 'wang', 'wan', 'yun', 'yve', 'wen', 'yuan', 'ai', 'ei', 'weng']

    leftList_1  = ['01' , '02', '03', '04' , '05', '06' , '07', '08', '09', '10', '11', '12', '13']
    leftLabel_1 = ['ang', 'er', 'wa', 'wei', 'wu', 'wai', 'ye', 'en', 'yu', 'wo', 'a' , 'ao', 'ou']

    # 右手
    rightList_0  = ['01', '03', '04', '06', '07', '08', '09', '10', '11', '12', '13']
    rightLabel_0 = ['ji', 'di', 'he', 'ni', 'ke', 'bi', 'pi', 'mi', 'ti', 'xi', 'zi']

    rightList_1  = ['01', '02', '03', '04' , '05', '06', '07', '08', '09', '10', '11', '12', '13']
    rightLabel_1 = ['yi', 'ge', 'de', 'shi', 'fu', 'le', 'qi', 'bu', 'pu', 'mu', 'te', 'ri', 'chi']

    # 本该右手在先，不知道为什么反了
    try:
        if handness == 'Left':
            if GuestureHand[1] == 0:
                code = leftLabel_0[leftList_0.index(GuestureHand[0])]
            else:
                code = leftLabel_1[leftList_1.index(GuestureHand[0])]
        else:
            if GuestureHand[1] == 0:
                code = rightLabel_0[rightList_0.index(GuestureHand[0])]
            else:
                code = rightLabel_1[rightList_1.index(GuestureHand[0])]
    except:
        print("err")
        code = ""

    return code.replace("xeng", "xing")

'''
def translate(Guesture: list) -> str:
    pass
'''
'''
a = Dataset(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])
a.makeDataset(1)
crossTest(np.asarray(a.trainSet), np.asarray(a.trainLabel))
#reconize_knn.knn_input(np.asarray(a.trainSet), np.asarray(a.trainLabel))
#reconize_knn.tree_input(np.asarray(a.trainSet), np.asarray(a.trainLabel))
#reconize_knn.Gaussian_input(np.asarray(a.trainSet), np.asarray(a.trainLabel))
'''
#test()
#print(test())
'''
sum = 0
for i in range(100):
    sum += test()

print(sum/100)
'''