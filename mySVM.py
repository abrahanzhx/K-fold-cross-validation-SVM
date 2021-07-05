# -*- coding: utf-8 -*-
# SVM Classification

# Importing the libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def cm_result(cm):
    # Calculate the accuracy of a confusion_matrix,where parameter 'cm' means confusion_matrix.
    a = cm.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred += cm[row, c]
            else:
                falsePred += cm[row, c]
    Accuracy = corrPred / (cm.sum())
    return Accuracy


if __name__ == '__main__':
    # Importing the dataset
    iris = load_iris()
    # Spliting the dataset in independent and dependent variables
    X = iris.data
    y = iris.target

    # # 方案一：拆成训练集和测试集
    # for g in [0.01, 0.1, 0.25, 1, 10, 100]:
    #     print('g=', g)
    #     # Splitting the dataset into the Training set and Test set
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
    #     # 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
    #     # Feature Scaling to bring the variable in a single scale
    #     sc = StandardScaler()
    #     X_train = sc.fit_transform(X_train)
    #     X_test = sc.transform(X_test)
    #     # Fitting SVC Classification to the Training set with linear kernel
    #     svcclassifier = SVC(kernel='rbf', random_state=0, C=1, gamma=g)
    #     svcclassifier.fit(X_train, y_train)
    #     # Predicting the Test set results using
    #     y_pred_train = svcclassifier.predict(X_train)
    #     # Predicting the Test set results using
    #     y_pred = svcclassifier.predict(X_test)
    #     # Making the Confusion Matrix
    #     cm_test = confusion_matrix(y_test, y_pred)
    #     cm_train = confusion_matrix(y_train, y_pred_train)
    #     print(cm_train)
    #     print(cm_test)
    #
    #     LinearTrainAccuracy = cm_result(cm_train)
    #     LinearTestAccuracy = cm_result(cm_test)
    #     print('Accuracy of the train is: ', round(LinearTrainAccuracy * 100, 2))
    #     print('Accuracy of the test is: ', round(LinearTestAccuracy * 100, 2))

    # 方案二：k折交叉验证
    for k in range(2, 11):
        print('k=', k)
        svcclassifier = SVC(kernel='rbf', decision_function_shape='ovo', gamma='auto')
        # cross_val_predict
        y_pred = cross_val_predict(svcclassifier, X, y, cv=k)
        cm_kFold = confusion_matrix(y, y_pred)
        print(cm_kFold)

        # lets see the actual and predicted value side by side
        y_compare = np.vstack((y, y_pred)).T
        # actual value on the left side and predicted value on the right hand side

        # finding accuracy from the confusion matrix.
        Accuracy = cm_result(cm_kFold)
        print ('Accuracy of the SVC Clasification is: ', Accuracy)
