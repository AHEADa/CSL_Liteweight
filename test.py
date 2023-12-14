import pipe
import reconize_knn
import numpy as np
from sklearn import model_selection, preprocessing

dataset = pipe.Dataset(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])
dataset.makeDataset(1)
data = dataset.trainSet
label = dataset.trainLabel
print("")


from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold, LeavePOut
import  warnings



def crossTest(x, y):
    result = []
    for i in range(100):
        x_train, x_test, y_train, y_test = \
                    model_selection.train_test_split(x, y, test_size = 0.1)
        reconize_knn.knn_input(x_train, y_train)
        #reconize_knn.tree_input(x_train, y_train)
        #reconize_knn.Gaussian_input(x_train, y_train)
        
        result.append(np.mean(y_test == reconize_knn.knn.predict(x_test)))
        #result.append(np.mean(y_test == reconize_knn.dt.predict(x_test)))
        #result.append(np.mean(y_test == reconize_knn.gnb.predict(x_test)))

    print("svm classifier accuacy:")
    print(np.mean(result))
 


# Z-Score标准化
zscore_scaler=preprocessing.StandardScaler()
data_scaler_1=zscore_scaler.fit_transform(data)
# Max-Min标准化
minmax_scaler=preprocessing.MinMaxScaler()
data_scaler_2=minmax_scaler.fit_transform(data)
# MaxAbs标准化
maxabs_scaler=preprocessing.MaxAbsScaler()
data_scaler_3=maxabs_scaler.fit_transform(data)
# RobustScaler准化
robust_scaler=preprocessing.RobustScaler()
data_scaler_4=robust_scaler.fit_transform(data)


crossTest(data_scaler_1, label)

'''

warnings.filterwarnings('ignore')


strKFold = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)

loout = LeavePOut(3)

#scores = cross_val_score(reconize_knn.knn, data_scaler_4, label, cv=strKFold)
scores = cross_val_score(reconize_knn.knn, data_scaler_2, label, cv=loout)
print("leave-one-out cross validation scores:{}".format(scores))
print("Mean score of leave-one-out cross validation:{:.2f}".format(scores.mean()))

'''