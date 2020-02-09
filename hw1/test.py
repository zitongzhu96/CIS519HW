#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:44:57 2019

@author: zhuzitong
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def getAccuracy(first,second):
    same=np.equal(first,second)
    top=np.sum(same)
    return (top/np.shape(first)[0])

def train_and_evaluate_sgd(X_train, y_train, X_test, y_test):
    SGD_model = SGDClassifier(loss='log', max_iter=10000)
    model = SGD_model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    accuracy1=getAccuracy(y_train_predict, y_train)
    y_test_predict = model.predict(X_test)
    accuracy2=getAccuracy(y_test_predict, y_test)
    return accuracy1, accuracy2
    
    
def train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test):
    rows=np.shape(X_train)[0]
    cols=np.shape(X_train)[1]
    stumps=[]
    features=[]
    stumpTree=np.zeros((rows,50))
    treeModel = DecisionTreeClassifier(criterion='entropy',max_depth=4)
    SGDModel = SGDClassifier(loss='log', max_iter=50000)
    for i in range(50): 
        randFeatures=np.random.choice(cols,size=int(cols/2),replace=False)
        features.append(randFeatures)
        half_X_train=X_train[:,randFeatures]
        stump=treeModel.fit(half_X_train,y_train)
        stumps.append(stump)
        stump_predict=stump.predict(half_X_train)
        stumpTree[:,i]=stump_predict
    testOutput=np.zeros((np.shape(X_test)[0],50))
    for j in range(50):
        half_X_test=X_test[:,features[j]]
        testOutput[:,j]=stumps[j].predict(half_X_test)
    FinalModel=SGDModel.fit(stumpTree,y_train)
    y_train_predict=FinalModel.predict(stumpTree)
    accurancy1=getAccuracy(y_train_predict, y_train)
    y_test_predict=FinalModel.predict(testOutput)
    accurancy2=getAccuracy(y_test_predict, y_test)
    return accurancy1, accurancy2
    

    

        
X_train = np.load('/Users/zhuzitong/Desktop/CIS519/hw1/hw1-materials/madelon/train-X.npy')
y_train = np.load('/Users/zhuzitong/Desktop/CIS519/hw1/hw1-materials/madelon/train-y.npy')
X_test = np.load('/Users/zhuzitong/Desktop/CIS519/hw1/hw1-materials/madelon/test-X.npy')
y_test = np.load('/Users/zhuzitong/Desktop/CIS519/hw1/hw1-materials/madelon/test-y.npy')
print(train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test))
print(train_and_evaluate_sgd(X_train, y_train, X_test, y_test))


    
      
    
    
    
