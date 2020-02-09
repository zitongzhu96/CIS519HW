#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:44:57 2019

@author: zhuzitong
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from string import ascii_lowercase

#---------Functions for Question 2 Part A-----------
#SGD 
def train_and_evaluate_sgd(X_train, y_train, X_test, y_test):
    SGD_model = SGDClassifier(loss='log', max_iter=10000)
    model = SGD_model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    accuracy1=accuracy_score(y_train_predict, y_train)
    accuracy2=accuracy_score(y_test_predict, y_test)
    return accuracy1, accuracy2

#Decision Tree
def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    tree_model = DecisionTreeClassifier(criterion='entropy')
    model = tree_model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    accuracy1=accuracy_score(y_train_predict, y_train)
    accuracy2=accuracy_score(y_test_predict, y_test)
    return accuracy1, accuracy2

#Decision Stump of Depth 4:
def train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test):
    stump_model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model = stump_model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    accuracy1=accuracy_score(y_train_predict, y_train)
    accuracy2=accuracy_score(y_test_predict, y_test)
    return accuracy1, accuracy2

#Decision Stumps as Features:  
def train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test):
    #Find the size of matrix
    rows=np.shape(X_train)[0]
    cols=np.shape(X_train)[1]
    #Build Stump Tree
    stumpTree=np.zeros((rows,50))
    #Build models:
    treeModel = DecisionTreeClassifier(criterion='entropy',max_depth=4)
    SGDModel = SGDClassifier(loss='log', max_iter=50000)
    #Build test output
    testOutput=np.zeros((np.shape(X_test)[0],50))
    
    for i in range(50): #For each stump:
        #Choose random half of features for train set
        randFeatures=np.random.choice(cols,size=int(cols/2),replace=False)
        half_X_train=X_train[:,randFeatures]
        #Train stump by decision tree model
        stump=treeModel.fit(half_X_train,y_train)
        #Predict by stump model
        stump_predict=stump.predict(half_X_train)
        #Put stump in the stump tree
        stumpTree[:,i]=stump_predict
        #Run for test set
        half_X_test=X_test[:,randFeatures]
        testOutput[:,i]=stump.predict(half_X_test)
    #Use SGD on stump tree
    FinalModel=SGDModel.fit(stumpTree,y_train)
    #Calculate accurancy for train and test sets.
    y_train_predict=FinalModel.predict(stumpTree)
    accurancy1=accuracy_score(y_train_predict, y_train)
    y_test_predict=FinalModel.predict(testOutput)
    accurancy2=accuracy_score(y_test_predict, y_test)
    return accurancy1, accurancy2

#The method that given to construct plot
def plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc, sgd_heldout_std, sgd_test_acc,
                 dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,
                 dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,
                 stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc):
    """
    Plots the final results from problem 2. For each of the 4 classifiers, pass
    the training accuracy, training standard deviation, held-out accuracy, held-out
    standard deviation, and testing accuracy.

    Although it should not be necessary, feel free to edit this method.
    """
    train_x_pos = [0, 4, 8, 12]
    cv_x_pos = [1, 5, 9, 13]
    test_x_pos = [2, 6, 10, 14]
    ticks = cv_x_pos

    labels = ['sgd', 'dt', 'dt4', 'stumps (4 x 50)']

    train_accs = [sgd_train_acc, dt_train_acc, dt4_train_acc, stumps_train_acc]
    train_errors = [sgd_train_std, dt_train_std, dt4_train_std, stumps_train_std]

    cv_accs = [sgd_heldout_acc, dt_heldout_acc, dt4_heldout_acc, stumps_heldout_acc]
    cv_errors = [sgd_heldout_std, dt_heldout_std, dt4_heldout_std, stumps_heldout_std]

    test_accs = [sgd_test_acc, dt_test_acc, dt4_test_acc, stumps_test_acc]

    fig, ax = plt.subplots()
    ax.bar(train_x_pos, train_accs, yerr=train_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='train')
    ax.bar(cv_x_pos, cv_accs, yerr=cv_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='held-out')
    ax.bar(test_x_pos, test_accs, align='center', alpha=0.5, capsize=10, label='test')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title('Models')
    ax.yaxis.grid(True)
    ax.legend()
    plt.tight_layout() 

#Method to get accurancies for cross-validation
def getAccurancy():
    SGD_train=[]
    DT_train=[]
    DTS_train=[]
    SGDS_train=[]
    SGD_heldout=[]
    DT_heldout=[]
    DTS_heldout=[]
    SGDS_heldout=[]
    for num in range(5):
        x_heldout_num = np.load('madelon/cv-heldout-X.'+str(num)+'.npy')
        y_heldout_num = np.load('madelon/cv-heldout-y.'+str(num)+'.npy')
        x_train_num=np.load('madelon/cv-train-X.'+str(num)+'.npy')
        y_train_num=np.load('madelon/cv-train-y.'+str(num)+'.npy')
        SGDmodel=train_and_evaluate_sgd(x_train_num, y_train_num,x_heldout_num, y_heldout_num)
        DTmodel=train_and_evaluate_decision_tree(x_train_num, y_train_num,x_heldout_num, y_heldout_num)
        DTSmodel=train_and_evaluate_decision_stump(x_train_num, y_train_num,x_heldout_num, y_heldout_num)
        SGDSmodel=train_and_evaluate_sgd_with_stumps(x_train_num, y_train_num,x_heldout_num, y_heldout_num)
        SGD_train.append(SGDmodel[0])
        DT_train.append(DTmodel[0])
        DTS_train.append(DTSmodel[0])
        SGDS_train.append(SGDSmodel[0])
        SGD_heldout.append(SGDmodel[1])
        DT_heldout.append(DTmodel[1])
        DTS_heldout.append(DTSmodel[1])
        SGDS_heldout.append(SGDSmodel[1])        
    return SGD_train,SGD_heldout, DT_train,DT_heldout, DTS_train,DTS_heldout, SGDS_train,SGDS_heldout

#Function to get confidence interval
def getConfidenceInterval(mean, std,n,t):
    return mean-t*std/np.sqrt(n),mean+t*std/np.sqrt(n)

#-------Functions for Question 2 Part B---------
def compute_features(names):  
    features=np.zeros((len(names),260))
    for i in range(len(names)):
        name=names[i].lower()
        firstname,lastname=name.split(' ')
        if len(firstname)>5:
            firstname=firstname[0:5]
        nums = np.array([(ord(char)-ord('a')) for char in firstname])
        nums+=np.arange(0,len(nums)*26,26)
        features[i,nums]=1
        if len(lastname)>5:
            lastname=lastname[0:5]
        nums = np.array([(ord(char)-ord('a')) for char in lastname])
        nums +=np.arange(5*26,(5+len(nums))*26,26)
        features[i,nums]=1
    return features

def transModel(path,model):
    names=[]
    with open(path,"r") as f:
        for line in f:
            line=line.strip()
            names.append(line)
    return model(names)

#——————————Extra Credit——————————
def compute_features2(names):  
    #Set vowel
    vowel = ['a','e','i','o','u']
    features=np.zeros((len(names),30))
    for i in range(0,len(names)):
        name=names[i].lower()
        firstname,lastname=name.split(' ')
        if len(firstname)>15:
            firstname=firstname[0:15]
        if len(lastname)>15:
            lastname=lastname[0:15]
        for j in range(0,min(len(firstname),15)):
            if firstname[j] in vowel:
                features[i][j] = 1
        for j in range(0,min(len(lastname),15)):
            if lastname[j] in vowel:
                features[i][j+15] = 1
    return features
#-----main function for Part A---------
def main1():
    #load model
    X_train = np.load('madelon/train-X.npy')
    y_train = np.load('madelon/train-y.npy')
    X_test = np.load('madelon/test-X.npy')
    y_test = np.load('madelon/test-y.npy')
    #Get accurancies
    dataset=getAccurancy()
    
    #get mean, std for each algorithm in train, test, and heldout set
    #-----sgd-----
    sgd_train_acc=np.mean(dataset[0])
    sgd_train_std=np.std(dataset[0])
    sgd_train_std1=np.std(dataset[0],ddof=1)
    
    sgd_heldout_acc=np.mean(dataset[1])
    sgd_heldout_std=np.std(dataset[1])
    sgd_heldout_std1=np.std(dataset[1],ddof=1)
    
    sgd_test_acc=train_and_evaluate_sgd(X_train, y_train, X_test, y_test)[1]
    #-----Decision Tree-----
    dt_train_acc=np.mean(dataset[2])
    dt_train_std=np.std(dataset[2])
    dt_train_std1=np.std(dataset[2],ddof=1)
    
    dt_heldout_acc=np.mean(dataset[3])
    dt_heldout_std=np.std(dataset[3])
    dt_heldout_std1=np.std(dataset[3],ddof=1)
    
    dt_test_acc=train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test)[1]
    
    #-----Decision Tree with depth 4-----
    dt4_train_acc=np.mean(dataset[4])
    dt4_train_std=np.std(dataset[4])
    dt4_train_std1=np.std(dataset[4],ddof=1)
    
    dt4_heldout_acc=np.mean(dataset[5])
    dt4_heldout_std=np.std(dataset[5])
    dt4_heldout_std1=np.std(dataset[5],ddof=1)
    
    dt4_test_acc=train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test)[1]
    
    #-----Stumps Tree-----
    stumps_train_acc=np.mean(dataset[6])
    stumps_train_std=np.std(dataset[6])
    stumps_train_std1=np.std(dataset[6],ddof=1)
    
    stumps_heldout_acc=np.mean(dataset[7])
    stumps_heldout_std=np.std(dataset[7])
    stumps_heldout_std1=np.std(dataset[7],ddof=1)
    
    stumps_test_acc=train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test)[1]
    
    #Load plot for these data
    plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc, sgd_heldout_std, sgd_test_acc,
                     dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,
                     dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,
                     stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc)
    
    #Set data for Confidence Interval, and print:
    n=5
    t=2.776
    print("Confidence Interval for sgd train: "+str(getConfidenceInterval(sgd_train_acc, sgd_train_std1,n,t)))
    print("Confidence Interval for sgd heldout: "+str(getConfidenceInterval(sgd_heldout_acc, sgd_heldout_std1,n,t)))
    print("Confidence Interval for decision tree train: "+str(getConfidenceInterval(dt_train_acc, dt_train_std1,n,t)))
    print("Confidence Interval for decision tree heldout: "+str(getConfidenceInterval( dt_heldout_acc, dt_heldout_std1,n,t)))
    print("Confidence Interval for decision tree(depth 4) train: "+str(getConfidenceInterval(dt4_train_acc,dt4_train_std1,n,t)))
    print("Confidence Interval for decision tree(depth 4) heldout: "+str(getConfidenceInterval(dt4_heldout_acc, dt4_heldout_std1,n,t)))
    print("Confidence Interval for sgd with stumps train: "+str(getConfidenceInterval(stumps_train_acc, stumps_train_std1,n,t)))
    print("Confidence Interval for sgd with stumps heldout: "+str(getConfidenceInterval(stumps_heldout_acc, stumps_heldout_std1,n,t)))



#--------Main Function for Question 2 PartB----------
def main2(model):
    #load data
    x_train=transModel("badges/train.names.txt",compute_features)
    x_test=transModel("badges/test.names.txt",compute_features)
    y_train=np.load("badges/train.labels.npy")
    y_test=np.load("badges/test.labels.npy")
    #Train model and get accurancy
    SGDmodel=train_and_evaluate_sgd(x_train, y_train,x_test, y_test)
    DTmodel=train_and_evaluate_decision_tree(x_train, y_train,x_test, y_test)
    DTSmodel=train_and_evaluate_decision_stump(x_train, y_train,x_test, y_test)
    SGDSmodel=train_and_evaluate_sgd_with_stumps(x_train, y_train,x_test, y_test)    
    print(SGDmodel,DTmodel,DTSmodel,SGDSmodel)

def main3():
    x_train=transModel("badges/train.names.txt",compute_features2)
    y_train=np.load("badges/train.labels.npy")
    test=transModel("badges/hidden-test.names.txt",compute_features2)
    stump_model = tree_model = DecisionTreeClassifier(criterion='entropy')
    model = stump_model.fit(x_train, y_train)
    np.savetxt('labels.txt',model.predict(test),'%i')

    
if __name__ == '__main__':
    main1()
    main2(compute_features)
    main3()
    
      
    
    
    
