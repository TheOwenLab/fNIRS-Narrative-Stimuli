# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:01:15 2022

@author: Matthew
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
import numpy.matlib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score,balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron,LogisticRegression,RidgeClassifier
from statsmodels.stats.multitest import fdrcorrection
from sklearn.inspection import permutation_importance
import pingouin as pg
import seaborn as sns
import xgboost as xgb
from scipy.io import loadmat,savemat
from scipy.stats import sem

nSub = 15; nCond = 2; nChan = 121; nHb = 2; SSList = [7, 28, 51, 65, 74, 91, 111, 124]
condIdx = [2,3]
# change directory
path = r"D:\Data\NaciRep\HealthyFinalAnalysis"
os.chdir(path)
file = "TSub.mat"; fileh = "TSubML.mat"

# load group mask
gMask = loadmat(os.path.join(path,"TKNvScramMaskFDR.mat"))["TakenMask"].T.reshape(-1,) 
nmask = loadmat(os.path.join(path,"FPMask.mat"))["FPmask"].T.reshape(-1,) 

# load in data
ISC = loadmat(os.path.join(path,file))["ISCResAverageT"]
ISCML = loadmat(os.path.join(path,fileh))["ISCResAverageT"]

# channel names
#cName = loadmat(os.path.join("IntactvScramNamesFDR.mat"))["sigRegions"]
#cName = loadmat(os.path.join("FPMaskNames"))["sigRegions"]
cName = pd.read_excel("ChannelProjToCortex.xlsx")["Label Name"].values
fpmask = nmask.astype("bool");
  
  
ISC = ISC[:,:,:]; ISCML = ISCML[:,:,:,:]                         
cNameArr = np.array(cName)
# remove short channels
cNameArr = np.delete(cNameArr,SSList)
nChan = ISC.shape[1]


# unpack data
# for each subject
datadf = []; dataX = []
for ss in range(0,nSub):
    td = ISC[:,:,ss]
    # and for each condition
    for nc in condIdx:
        # and for each hb type
        tdc = td[nc,:]
        # store this in ML array
        iscTemp = tdc.tolist()
        dataX.append(np.array([nc,ss] + iscTemp))
        # make other columns for plotting
        tArr = np.ones((nChan,4)) * np.nan
        tArr[:,0] = iscTemp
        # also collect subject, hb, condition and channel info
        tArr[:,1] = np.ones_like(iscTemp) * ss; tArr[:,2] = np.ones_like(iscTemp) * nc; 
        tArr[:,3] = np.arange(0,nChan,1)
        datadf.append(tArr)

            
X = np.stack(dataX); df = pd.DataFrame(np.concatenate(datadf),\
                                       columns = ["ISC","Subject","Condition","Channel"]) 

# initalize feature importance lists
featureImport = []; lfeatureImport = []; pfeatureImport = [];
xgfimport = []

# combine all relavent info
df["ChannelName"] = cNameArr.tolist() * nSub * nCond
df["Subject"] = df["Subject"] + 1

# set up np array for ML
Xs = X[:,1:]; Y = X[:,0]

# uppack ML ISC 
# unpack data
# for each subject
dataX = []
for ss in range(0,nSub):
    td = ISCML[:,:,ss,:]
    # and for each condition
    tdc = np.concatenate([td[2,:,:],td[3,:,:]],axis = 1)    
    iscTemp = tdc
    dataX.append(iscTemp)

xML = np.stack(dataX); yML = np.repeat([2,3],repeats = (nSub - 1))

# or load FDR mask from MATLAB
FDRMask = loadmat("TakenMaskFDRML.mat")['TKNMask'].astype("bool")
sigMask = loadmat("TakenMaskML.mat")['TKNMask'].astype("bool")

# # remove significant regions that could be due to chance
discsRemMask = np.where(sigMask.sum(axis = 1) <= 5)
discRemMask = np.where(FDRMask.sum(axis = 1) <= 3)
for rm in range(0,len(discsRemMask)):
    sigMask[discsRemMask[rm],:] = False  
for rm in range(0,len(discRemMask)):
    FDRMask[discRemMask[rm],:] = False 
# remove non FP channels if appropriate
nonFP = np.where(nmask == 0)[0];
for nfp in range(0,len(nonFP)):
    sigMask[nonFP[nfp],:] = False
    FDRMask[nonFP[nfp],:] = False



# generate participant randomization
nIter = 1; nRand = 14;
CondList = set([2,3])

nSubList = np.array(range(0,nSub-1))
np.random.seed(59)
groupPerf = []; subPerf = [];

gProb = np.ones((nSub,nIter,nCond,nCond,nChan)) * np.nan # subject iter, true label, pred
gAcc = np.ones((nSub,nIter,nChan))

for ss in range(0,nSub):
    # get subject data and set as train set
    print(ss)
    
    for ii in range(nIter):
        # for each channel
        for cc in range(0,nChan):
            X_test = Xs[Xs[:,0] == ss,1:]; y_test = Y[Xs[:,0] == ss]
            X_train = xML[ss,:,:].T; y_train = yML
            
            # partial out subset of training set each iteration
            #ridx = np.random.permutation(nSubList[nSubList != ss])[0:nRand]
            ridx = np.random.permutation(nSubList)[0:nRand]
            iterIdx = np.concatenate([ridx,ridx + (nSub - 1)])
            X_train = X_train[iterIdx,cc]; y_train = y_train[iterIdx]
            X_test = X_test[:,cc]
            # randomize the arrays
            trainrand = np.random.permutation(y_train.shape[0]); testrand = np.random.permutation(y_test.shape[0]);
            X_test = X_test[testrand]; y_test = y_test[testrand];
            X_train = X_train[trainrand]; y_train = y_train[trainrand]
            clf = LazyClassifier(predictions=True)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            predictions = predictions.drop(labels = ["DummyClassifier","LGBMClassifier"], axis = 1)
            gPred = predictions.apply(lambda x: x.value_counts(),axis = 1).fillna(0)/predictions.shape[1]
    
            if gPred.shape[1] < 2:
                # find the missing condition label
                cSet = set(gPred.columns)
                mCond = list(CondList.difference(cSet))
                # set the missing condition label(s) to 0
                gPred[mCond] = 0;
                # reorder dataframe
                gPred = gPred[[2,3]]
                # save to array
                gProb[ss,ii,y_test.astype("int") - 2,:,cc] = gPred
                
            else:
                gPred = gPred[[2,3]]
                gProb[ss,ii,y_test.astype("int") - 2,:,cc] = gPred
    
            #print(y_test)
            #print(gPred)
            accPred = gPred.idxmax(axis = 1).values            
            gAcc[ss,ii,cc] = balanced_accuracy_score(y_test,accPred)    
            models["Subject"] = ss + 1
            subPerf.append(models)


# just quick summary plots
gAcc = gAcc[:,0,:]; gProb = gProb[:,0,:,:,:] 

# a predictive channel is one that is accurate across particpants
# mean accuracy for each channel
pAcc = np.mean(gAcc,axis = 0)
# and is confident across participants
pProb = np.mean(gProb,axis = 0)
# then average across the diagnonal 
pConf = (pProb[0,0,:] + pProb[1,1,:])/2

# save both confidence and mask
savemat("TakenSingleChannelRes.mat",{"pConf":pConf, "pAcc":pAcc})