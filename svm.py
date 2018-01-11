# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 03:13:26 2017

@author: JAY SHAH
"""

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
style.use('ggplot')
import random



#Dataset is split into two parts from here (Training and Test Dataset)
def splitDataset1(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    testSet = []
    copy = dataset
    i=0
    while i != len(copy):
        #index = random.randrange(len(copy))
        if len(trainSet) < trainSize:
            trainSet.append(copy.iloc[i])
        else:
            testSet.append(copy.iloc[i])
        i = i+1
      
    #print trainSet
    #print testSet
    return [trainSet, testSet]

def splitDataset(dataset,dataset1):
    #trainSize = int(len(dataset) * splitRatio)
    #print dataset1
    trainSet1 = []
    trainSet2 = []
    i=0
    value = dataset['clicks']
    while i < len(dataset1):
        if value.iloc[i] == 0:
            #print "yeee"
            trainSet1.append(dataset1[i])
        else:
            #print "ahaaa"
            trainSet2.append(dataset1[i])
        i= i+1
                  
    return [trainSet1,trainSet2]


class Support_Vector_Machine:
    # train
    def fit(self, data):
        self.data = data
        #print self.data
        # { ||w||: [w,b] }
        opt_dict = {}
        
        #Array of transformation vector is defined, helpful in trying out combinations of W and b
        transforms = [[1,1,1,1,1,1,1,1,1,1,1,1],
                       [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1],
                       [-1,-1,-1,-1,1,1,1,1,-1,-1,1,1],
                       [-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1]
                      ]

        total_data = []
        for yi in self.data:
            for features in self.data[yi]:
                for featurevalue in features:
                    total_data.append(featurevalue)

        self.max_f_value = max(total_data)
        self.min_f_value = min(total_data)
        total_data = None
        print self.max_f_value
        print self.min_f_value
        # support vectors yi(xi.w+b) = 1
        
        #Again Defining Step Sizes array which will decrease the value of W
        stepsizes = [self.max_f_value * 0.1,
                      self.max_f_value * 0.01,
                      ]

        
        
        # Very Costly
        b_range_multiple = 5
        
        # with b as we do w
        b_multiple = 2
        latest_optimum = self.max_f_value*10
        
        for step in stepsizes:
            w = np.array([latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_f_value*b_range_multiple),
                                   self.max_f_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    #print "Hello"
                                    #print yi*(np.dot(w_t,xi)+b)
                                    found_option = False
                                else:
                                    #print "Gayelo"
                                    #print yi*(np.dot(w_t,xi)+b)
                                    found_option = True
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                        
                        if found_option:    
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
            #print opt_dict
            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                #print(xi,':',yi*(np.dot(self.w,xi)+self.b)) 
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                
                #print(xi,':',yi*(np.dot(self.w,xi)+self.b))            

    def predict(self,features):
        # sign( x.w+b )
        #print self.w     #Value of W for which data will be predicted
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        print classification  # -1 means ad will not clicked and 1 means ad will be clicked

    
        
file = 'train.csv'
#file = 'D:/Project_Data_Mining/random.csv'

dataset1 = pd.read_csv(file)
dataset2 = pd.read_csv(file).drop('clicks',axis=1)
cset1 = dataset2[['C1','banner_pos','device_type','device_conn_type','C14',
 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']]
mean1 = cset1.mean()
std1 = cset1.std()
normalised1 = (cset1 - mean1)/std1   #Normalising the Input Data
#print normalised1

trainingset, testset = splitDataset1(normalised1,0.80)  #Giving 80% of data to trainset and 20% to testset
trainingset1, trainingset2 = splitDataset(dataset1,trainingset) #Training set is splitted into two parts according to its class 0 or 1



dataset1 = np.array(trainingset1)
dataset2 = np.array(trainingset2)

data_dict = {-1:dataset1,
             1:dataset2}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)



for p in trainingset:
    svm.predict(p)
