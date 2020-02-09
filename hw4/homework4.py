#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:27:49 2019

@author: zhuzitong
"""

import json
import numpy
from scipy.special import logsumexp
from collections import defaultdict
import random

# You should not need to edit this cell
def load_dataset(file_path):
    D = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            D.append(instance['document'])
            y.append(instance['label'])
    return D, y

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X

def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    # TODO
    wordSet=set()
    vocSet=set()
    vocSet.add('<unk>')
    for document in D:
        for token in document:
            if token in wordSet:
                vocSet.add(token)
            wordSet.add(token)
    return vocSet

class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        features=dict()
        for d in doc:
            if d in vocab:
                features[d]=1
            else:
                features['<unk>']=1
        return features
    
class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        features=defaultdict(int)       
        for word in doc:
            if word in vocab:
                features[word]=features[word]+1
            else:
                features['<unk>']=features['<unk>']+1
        return dict(features)

def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    # TODO
    idf_dict=dict()
    D_dictList=[]
    for index in range(len(D)):
        temp_dict=dict()
        for word in D[index]:
            if word in vocab:
                temp_dict[word]=1
            else:
                temp_dict['<unk>']=1
        D_dictList.append(temp_dict)
    for index in range(len(D)):
        for key in D_dictList[index]:
            if idf_dict.get(key):
                idf_dict[key]+=1
            else:
                idf_dict[key]=1
    for word in idf_dict:
        idf_dict[word]=numpy.log(len(D)/idf_dict[word])
    return idf_dict
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        word_dict = {}
        num_word_dict=defaultdict(int)       
        for word in doc:
            if word in vocab:
                num_word_dict[word]=num_word_dict[word]+1
            else:
                num_word_dict['<unk>']=num_word_dict['<unk>']+1
        num_word_dict=dict(num_word_dict)
        for i in doc:
            if i in num_word_dict:
                word_dict[i] = num_word_dict[i]* self.idf[i]
            if i not in num_word_dict:
                word_dict['<unk>'] = num_word_dict['<unk>'] * self.idf['<unk>']
        return word_dict
    
def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    # TODO
    p_y=dict()
    p_v_y=dict()
    p_y[0]=0
    p_y[1]=0
    p_v_y[0]=dict()
    p_v_y[1]=dict()
    sum_fdw_0=0
    sum_fdw_1=0
    index_dict=dict()
    index_dict[0]=[]
    index_dict[1]=[]
    for i in range(len(y)):
        if y[i]==0:
            p_y[0]+=1
            index_dict[0].append(i)
        else:
            p_y[1]+=1
            index_dict[1].append(i)
    p_y[0]=p_y[0]/len(y)  
    p_y[1]=p_y[1]/len(y)    
    for i in index_dict:
        for j in index_dict[i]:             
            for key in X[j]:
                if p_v_y[i].get(key):
                    p_v_y[i][key]=p_v_y[i][key]+X[j][key]

                else:
                    p_v_y[i][key]=X[j][key]
                if i==0:                  
                    sum_fdw_0+=X[j][key]
                else:
                    sum_fdw_1+=X[j][key]
    d1=k*len(vocab)+sum_fdw_0
    d2=k*len(vocab)+sum_fdw_1
    for word in p_v_y[0]:
        p_v_y[0][word]=(p_v_y[0][word]+k)/d1
    for word in p_v_y[1]:
        p_v_y[1][word]=(p_v_y[1][word]+k)/d2
    for v in vocab:
        if v not in p_v_y[0]:
            p_v_y[0][v]=k/d1
        if v not in p_v_y[1]:
            p_v_y[1][v]=k/d2             
    return p_y, p_v_y

def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    # TODO
    predictions=[]
    confidences=[]
    for d in D:
        sum_pvy_0=0
        sum_pvy_1=0
        for word in d:
            if word in p_v_y[1]:
                sum_pvy_0+=numpy.log(p_v_y[0][word])
                sum_pvy_1+=numpy.log(p_v_y[1][word])
            else:
                sum_pvy_0+=numpy.log(p_v_y[0]['<unk>'])
                sum_pvy_1+=numpy.log(p_v_y[1]['<unk>'])
        sum_pvy_0+=numpy.log(p_y[0])
        sum_pvy_1+=numpy.log(p_y[1])
        confidence=0
        p_d=[sum_pvy_0,sum_pvy_1]
        p_d=logsumexp(numpy.array(p_d))
        if sum_pvy_0>sum_pvy_1:
            predictions.append(0)
            confidence=numpy.exp(sum_pvy_0-p_d)
        else:
            predictions.append(1)
            confidence=numpy.exp(sum_pvy_1-p_d)
        confidences.append(confidence)
    return predictions, confidences
            

def train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, D_valid, y_valid, k, vocab, mode):
    """
    Trains the Naive Bayes classifier using the semi-supervised algorithm.
    
    X_sup: A list of the featurized supervised documents.
    y_sup: A list of the corresponding supervised labels.
    D_unsup: The unsupervised documents.
    X_unsup: The unsupervised document representations.
    D_valid: The validation documents.
    y_valid: The validation labels.
    k: The smoothing parameter for Naive Bayes.
    vocab: The vocabulary as a set of tokens.
    mode: either "threshold" or "top-k", depending on which selection
        algorithm should be used.
    
    Returns the final p_y and p_v_y (see `train_naive_bayes`) after the
    algorithm terminates.    
    """
    # TODO
    p_y1,p_v_y1=train_naive_bayes(X_sup, y_sup, k, vocab)
    res=predict_naive_bayes(D_valid, p_y1, p_v_y1)
    acc=0
    for index in range(len(res[0])):
        if res[0][index]==y_valid[index]:
            acc+=1
    print("Initial accuracy: "+str(acc/len(res[0])))
    while True:
        p_y,p_v_y=train_naive_bayes(X_sup, y_sup, k, vocab)
        U_y,P_y=predict_naive_bayes(D_unsup, p_y, p_v_y)
        S_x_new=[]
        S_y_new=[]   
        Us=[]
        for i in range(len(X_unsup)):
            Us.append((X_unsup[i], U_y[i],P_y[i],D_unsup[i]))
        Us = sorted(Us, key = lambda p: p[2], reverse = True)
        if mode=='threshold':
            for i in range(len(Us)):
                if Us[i][2]>=0.98:
                    S_x_new.append(Us[i][0])
                    S_y_new.append(Us[i][1])
                    X_sup.append(Us[i][0])
                    y_sup.append(Us[i][1])
                    X_unsup.remove(Us[i][0])
                    D_unsup.remove(Us[i][3])
                else:
                    break
        if mode=='top-k':
            for i in range(10000):
                if len(Us)>=10000:
                    S_x_new.append(Us[i][0])
                    S_y_new.append(Us[i][1])
                    X_sup.append(Us[i][0])
                    y_sup.append(Us[i][1])
                    X_unsup.remove(Us[i][0])
                    D_unsup.remove(Us[i][3])
                else:
                    if i==(len(Us)-1):
                        break
        if len(S_x_new)==0:
            return p_y,p_v_y
   

def representation_experiments():
    vocab = get_vocabulary(D_train)
    k_list=[0.001,0.01,0.1,1.0,10.0]
    print('Use B-Bow:')
    b_bow_res=dict()
    featurizer1 = BBoWFeaturizer()
    X1 = convert_to_features(D_train, featurizer1, vocab)
    for k in k_list: 
        p_y,p_v_y=train_naive_bayes(X1, y_train, k, vocab)
        res=predict_naive_bayes(D_valid, p_y, p_v_y)
        acc=0
        for index in range(len(res[0])):
            if res[0][index]==y_valid[index]:
                acc+=1
        b_bow_res[k]=acc/len(y_valid)
        print('k='+str(k)+': the accuracy is: '+str(b_bow_res[k]))
    best_k1=findBestKey(b_bow_res)
    print("The best k is:"+str(best_k1))
    
    p_y1,p_v_y1=train_naive_bayes(X1, y_train, k, vocab)
    res1=predict_naive_bayes(D_test, p_y1, p_v_y1)
    acc1=0
    for index in range(len(res1[0])):
        if res1[0][index]==y_test[index]:
            acc1+=1
    acc1=acc1/len(y_test)
    print('The accuracy in k='+str(best_k1)+': '+str(acc1))
    
    print('\nUse C-Bow')
    c_bow_res=dict()
    featurizer2 = CBoWFeaturizer()
    X2 = convert_to_features(D_train, featurizer2, vocab)
    for k in k_list: 
        p_y,p_v_y=train_naive_bayes(X2, y_train, k, vocab)
        res=predict_naive_bayes(D_valid, p_y, p_v_y)
        acc=0
        for index in range(len(res[0])):
            if res[0][index]==y_valid[index]:
                acc+=1
        c_bow_res[k]=acc/len(y_valid)
        print('k='+str(k)+': the accuracy is: '+str(c_bow_res[k]))
    best_k2=findBestKey(c_bow_res)
    print("The best k is:"+str(best_k2)) 
    
    p_y2,p_v_y2=train_naive_bayes(X2, y_train, k, vocab)
    res2=predict_naive_bayes(D_test, p_y2, p_v_y2)
    acc2=0
    for index in range(len(res2[0])):
        if res2[0][index]==y_test[index]:
            acc2+=1
    acc2=acc2/len(y_test)
    print('The accuracy in k='+str(best_k2)+': '+str(acc2))
    
    
    print('\nUse TF-IDF:')
    tf_idf_res=dict()
    idf = compute_idf(D_train, vocab)
    featurizer3 = TFIDFFeaturizer(idf)
    X3 = convert_to_features(D_train, featurizer3, vocab)
    for k in k_list: 
        p_y,p_v_y=train_naive_bayes(X3, y_train, k, vocab)
        res=predict_naive_bayes(D_valid, p_y, p_v_y)
        acc=0
        for index in range(len(res[0])):
            if res[0][index]==y_valid[index]:
                acc+=1
        tf_idf_res[k]=acc/len(y_valid)
        print('k='+str(k)+': the accuracy is: '+str(tf_idf_res[k]))
    best_k3=findBestKey(tf_idf_res)
    print("The best k is:"+str(best_k3))
    
    p_y3,p_v_y3=train_naive_bayes(X3, y_train, k, vocab)
    res3=predict_naive_bayes(D_test, p_y3, p_v_y3)
    acc3=0
    for index in range(len(res3[0])):
        if res3[0][index]==y_test[index]:
            acc3+=1
    acc3=acc3/len(y_test)
    print('The accuracy in k='+str(best_k3)+': '+str(acc3))
                 
def findBestKey(res_list):
    best_k=0
    maxAcc=0
    for k in res_list:
        if res_list[k]>maxAcc:
            best_k=k
            maxAcc=res_list[k] 
    return best_k

def semi_experiments(N,D,y,featurizer,vocab,mode):
    randList=random.sample(range(0,len(D)),N);
    X_sup=[]
    y_sup=[]
    D_unsup=D.copy()
    X=convert_to_features(D, featurizer, vocab)
    X_unsup=X.copy()
    for i in randList:
        X_sup.append(X[i])     
        y_sup.append(y[i])
        D_unsup.remove(D[i])
        X_unsup.remove(X[i])
    p_y,p_v_y=train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, X, y, 0.1,vocab,mode)
    res1=predict_naive_bayes(D_valid, p_y, p_v_y)
    acc1=0
    for index in range(len(res1[0])):
        if res1[0][index]==y_valid[index]:
            acc1+=1
    print("Final valid accuracy: "+str(acc1/len(res1[0])))
    res2=predict_naive_bayes(D_test, p_y, p_v_y)
    acc2=0
    for index in range(len(res2[0])):
        if res2[0][index]==y_test[index]:
            acc2+=1
    print("Final test accuracy: "+str(acc2/len(res2[0])))

D_train, y_train = load_dataset('/Users/zhuzitong/Desktop/hw4-materials/data/train.jsonl')
D_valid, y_valid = load_dataset('/Users/zhuzitong/Desktop/hw4-materials/data/valid.jsonl')
D_test, y_test = load_dataset('/Users/zhuzitong/Desktop/hw4-materials/data/test.jsonl')
vocab = get_vocabulary(D_train)

representation_experiments()
featurizer1 = BBoWFeaturizer()
featurizer2 = CBoWFeaturizer()
idf = compute_idf(D_train, vocab)
featurizer3 = TFIDFFeaturizer(idf)

print("Random Select 50:")
print("Threshold for BBow: ")
semi_experiments(50,D_train,y_train,featurizer1,vocab,"threshold")
print("Threshold for CBow: ")
semi_experiments(50,D_train,y_train,featurizer2,vocab,"threshold")
print("Threshold for TF-idF: ")
semi_experiments(50,D_train,y_train,featurizer3,vocab,"threshold")

print("Top-K for BBow: ")
semi_experiments(50,D_train,y_train,featurizer1,vocab,"top-k")
print("Top-K for CBow: ")
semi_experiments(50,D_train,y_train,featurizer2,vocab,"top-k")
print("Top-K for TF-idF: ")
semi_experiments(50,D_train,y_train,featurizer3,vocab,"top-k")

print("Random Select 500:")
print("Threshold for BBow: ")
semi_experiments(500,D_train,y_train,featurizer1,vocab,"threshold")
print("Threshold for CBow: ")
semi_experiments(500,D_train,y_train,featurizer2,vocab,"threshold")
print("Threshold for TF-idF: ")
semi_experiments(500,D_train,y_train,featurizer3,vocab,"threshold")
print("Top-K for BBow: ")
semi_experiments(500,D_train,y_train,featurizer1,vocab,"top-k")
print("Top-K for CBow: ")
semi_experiments(500,D_train,y_train,featurizer2,vocab,"top-k")
print("Top-K for TF-idF: ")
semi_experiments(500,D_train,y_train,featurizer3,vocab,"top-k")

print("Random Select 5000:")
print("Threshold for BBow: ")
semi_experiments(5000,D_train,y_train,featurizer1,vocab,"threshold")
print("Threshold for CBow: ")
semi_experiments(5000,D_train,y_train,featurizer2,vocab,"threshold")
print("Threshold for TF-idF: ")
semi_experiments(5000,D_train,y_train,featurizer3,vocab,"threshold")
print("Top-K for BBow: ")
semi_experiments(5000,D_train,y_train,featurizer1,vocab,"top-k")
print("Top-K for CBow: ")
semi_experiments(5000,D_train,y_train,featurizer2,vocab,"top-k")
print("Top-K for TF-idF: ")
semi_experiments(5000,D_train,y_train,featurizer3,vocab,"top-k")
 