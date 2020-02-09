#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:43:30 2019

@author: zhuzitong
"""
import numpy as np

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