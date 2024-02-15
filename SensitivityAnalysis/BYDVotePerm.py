# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:55:00 2023

@author: Admin
"""

# import helper function
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib
import pingouin as pg
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import sem
from joblib import dump, load


# import preprocessing 
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.metrics import classification_report, roc_auc_score,balanced_accuracy_score,confusion_matrix,recall_score,precision_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from hyperopt import fmin,tpe,hp,Trials
from sklearn.decomposition import PCA 

# import classifiers 
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,\
    StackingClassifier,RandomForestClassifier,VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import Perceptron,LogisticRegression,RidgeClassifier,RidgeClassifierCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier,RidgeClassifier,LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

nSub = 26; nCond = 2; nChan = 121; nHb = 2; SSList = [7, 28, 51, 65, 74, 91, 111, 124]
condIdx = [0,1]
# change directory
path = r'D:\Data\NaciRep\Sensitivity'
os.chdir(path)
file = "ISCRm.mat"; fileh = "ISCHoldOut.mat"

# load in data
ISC = loadmat(os.path.join(path,file))["ISC"]
ISCML = loadmat(os.path.join(path,fileh))["ISC"]

# average data across HbO and HbR and drop rest
ISC = np.mean(ISC[0:4,:,:,:],2)
ISCML = np.mean(ISCML,2)

# channel names
cName = pd.read_excel("ChannelProjToCortex.xlsx")["Label Name"].values                 
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

sns.catplot(data=df, x = "Condition", y = "ISC",hue = "Subject",kind = "box")
plt.show()
    
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
    tdc = np.concatenate([td[0,:,:],td[1,:,:]],axis = 1)    
    iscTemp = tdc
    # normalize iscTemp
    #iscTemp = (iscTemp - np.mean(iscTemp,0))/np.std(iscTemp,0)
    dataX.append(iscTemp)

xML = np.stack(dataX); yML = np.repeat([0,1],repeats = (nSub - 1))


# set Ys to 0 and 1
Y = Y == 0; yML = yML == 0;

# or load FDR mask from MATLAB
TKNMask = loadmat("GroupFDRML.mat")['GroupFDR'][0,:,:].astype("bool")
TKNSMask = loadmat("GroupFDRML.mat")['GroupFDR'][1,:,:].astype("bool")

# alternatively, use IgS mask
IgSMask = loadmat("FDRGroupBinIgS.mat")['FDRGroupBinIgS'][0,:,:].astype("bool")


FDRMask = TKNMask | TKNSMask

# generate participant randomization
nIter = 1; nRand = 25;
CondList = set([2,3])

nSubList = np.array(range(0,nSub-1))
np.random.seed(59)
groupPerf = []; subPerf = [];

gProb = np.ones((nSub,nIter,nCond,nCond)) * np.nan # subject iter, true label, pred
gAcc = np.ones((nSub,nIter))
tPred = np.zeros((nCond,nCond))

nFold = 3;  nPerm = 1000;
np.random.RandomState(23)
 
# load model classifier
MODELS = load("BYDMODELS")
ACC = np.zeros((nPerm,nSub)); PRECISION =  np.zeros((nPerm,nSub)); RECALL =  np.zeros((nPerm,nSub))
randcounter = 0;

for perm in range(0,nPerm):
    for ss in range(0,nSub):
        # get subject data and set as train set
        print(ss + 1)
        # get train and test sets
        X_test = Xs[Xs[:,0] == ss,1:]; y_test = Y[Xs[:,0] == ss]
        X_train = xML[ss,:,:].T; y_train = yML
        
        # take only areas within the mask
        X_train = X_train[:,FDRMask[:,ss].reshape(-1,)]
        X_test = X_test[:,FDRMask[:,ss].reshape(-1,)]
    
        # randomize the arrays
        trainrand = np.random.permutation(y_train.shape[0]); testrand = np.random.permutation(y_test.shape[0]);
        X_test = X_test[testrand,:]; y_test = y_test[testrand];
        X_train = X_train[trainrand,:]; y_train = y_train[trainrand]
        
        # randomly permute array
        y_test = np.random.permutation(y_test); y_train = np.random.permutation(y_train)
        
        # train and test each model
        cMODEL = MODELS[ss]; tMODELS = []
        WEIGHTS = []
        for model in cMODEL:
            # fit model
            model_name = model[0]
            model_pipe = model[1]
            model_pipe.fit(X_train,y_train)
            # store models
            tMODELS.append((model_name,model_pipe))
            # get weights
            WEIGHTS.append(model_pipe.score(X_train,y_train))
        
        # instantiate voting classifier
        vt = VotingClassifier(estimators = tMODELS,voting = "hard", weights = 1 + np.array(WEIGHTS))
        vt.fit(X_train,y_train)
        y_pred = vt.predict(X_test)
        # get accuracy metric
        acc = balanced_accuracy_score(y_test,y_pred)
        # store accuracy metric
        ACC[perm,ss] = acc
        # get precision metric
        pre = precision_score(y_test,y_pred)
        # store accuracy metric
        PRECISION[perm,ss] = pre
        # get recall metric
        rec = recall_score(y_test,y_pred)
        # store accuracy metric
        RECALL[perm,ss] = rec

np.save("BYDAccPerm.npy",ACC);
np.save("BYDPrecisionPerm.npy",PRECISION)
np.save("BYDRecallPerm.npy",RECALL)    