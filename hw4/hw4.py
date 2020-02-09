#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:27:05 2019

@author: zhuzitong
"""
import json
import numpy
from scipy.special import logsumexp
from collections import defaultdict
import random
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

