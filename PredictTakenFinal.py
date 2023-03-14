# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:29:42 2023

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
from scipy.io import loadmat
from scipy.stats import sem

nSub = 15; nCond = 2; nChan = 121; nHb = 2; SSList = [7, 28, 51, 65, 74, 91, 111, 124]
condIdx = [2,3]
# change directory
path = os.getcwd()
os.chdir(path)
file = "TSub.mat"; fileh = "TSubML.mat"

# load in data
ISC = loadmat(os.path.join(path,file))["ISCResAverageT"]
ISCML = loadmat(os.path.join(path,fileh))["ISCResAverageT"]

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
    tdc = np.concatenate([td[2,:,:],td[3,:,:]],axis = 1)    
    iscTemp = tdc
    dataX.append(iscTemp)

xML = np.stack(dataX); yML = np.repeat([2,3],repeats = (nSub - 1))

# or load FDR mask from MATLAB
FDRMask = loadmat("TakenMaskFDRML.mat")['TKNMask'].astype("bool")


# # remove significant regions that could be due to chance
discRemMask = np.where(FDRMask.sum(axis = 1) <= 3)
for rm in range(0,len(discRemMask)):
    FDRMask[discRemMask[rm],:] = False 


# replot difference between conditions
# plot values
# look at only mask
mIdx = np.matlib.repmat(FDRMask.sum(axis = 1) > 1,1,nSub); mIdx = np.matlib.repmat(mIdx,1,nCond)
sns.catplot(data=df[mIdx.reshape(-1,)==1], x = "Condition", y = "ISC",hue = "Subject",kind = "box")
plt.show()

# generate participant randomization
nIter = 1000; nRand = 10;
CondList = set([2,3])

nSubList = np.array(range(0,nSub-1))
np.random.seed(59)
groupPerf = []; subPerf = [];

gProb = np.ones((nSub,nIter,nCond,nCond)) * np.nan # subject iter, true label, pred
gAcc = np.ones((nSub,nIter))
tPred = np.zeros((nCond,nCond))

for ss in range(0,nSub):
    # get subject data and set as train set
    print(ss)
    
    for ii in range(nIter):
    
        X_test = Xs[Xs[:,0] == ss,1:]; y_test = Y[Xs[:,0] == ss]
        X_train = xML[ss,:,:].T; y_train = yML
        
        # partial out subset of training set each iteration
        #ridx = np.random.permutation(nSubList[nSubList != ss])[0:nRand]
        ridx = np.random.permutation(nSubList)[0:nRand]
        iterIdx = np.concatenate([ridx,ridx + (nSub - 1)])
        #unique, counts = np.unique(y_train[iterIdx], return_counts=True)
        #print(counts)
        X_train = X_train[iterIdx,:]; y_train = y_train[iterIdx]
                
        # take only areas within the mask
        X_train = X_train[:,FDRMask[:,ss].reshape(-1,)]
        X_test = X_test[:,FDRMask[:,ss].reshape(-1,)]

        # randomize the arrays
        trainrand = np.random.permutation(y_train.shape[0]); testrand = np.random.permutation(y_test.shape[0]);
        X_test = X_test[testrand,:]; y_test = y_test[testrand];
        X_train = X_train[trainrand,:]; y_train = y_train[trainrand]

        clf = LazyClassifier(predictions=True)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        predictions = predictions.drop(labels = ["DummyClassifier","LGBMClassifier"], axis = 1) # both classifiers only predict one label
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
            gProb[ss,ii,y_test.astype("int") - 2,:] = gPred
            
        else:
            gPred = gPred[[2,3]]
            gProb[ss,ii,y_test.astype("int") - 2,:] = gPred

        #print(y_test)
        #print(gPred)
        accPred = gPred.idxmax(axis = 1).values            
        gAcc[ss,ii] = balanced_accuracy_score(y_test,accPred)    
        models["Subject"] = ss + 1
        subPerf.append(models)


# plot mean results first
# average over iterations
mgProb = np.mean(gProb,axis = 1)
# average over group
groupProb = np.mean(mgProb,axis = 0)
# plot griup results

sns.set_theme(style = "white", rc={'figure.figsize': (4,3),"xtick.labelsize": 15, \
            'xtick.alignment':'center',"ytick.labelsize": 15,'font.family':'Arial','figure.dpi':500, \
            "axes.labelsize":15,"figure.facecolor":"white",'axes.facecolor':'white',"axes.spines.right": False,"axes.spines.left": False,\
                "axes.spines.top": False,"axes.spines.bottom": False})


sns.heatmap(groupProb, annot = True,cmap = "Purples",annot_kws={"fontsize":15})
plt.xticks([0.5,1.5], labels = ["Taken", "Taken Scrambled"])
plt.yticks([0.5,1.5], labels = ["Taken", "Taken Scrambled"])
plt.title("Taken vs Taken Scrambled Group Results")
plt.show()

# now plot each subs results

for ss in range(0,nSub):
    sns.heatmap(mgProb[ss,:,:], annot = True,cmap = "Purples",vmax=0.85,annot_kws={"fontsize":15})
    plt.xticks([0.5,1.5], labels = ["Taken", "Taken Scrambled"])
    plt.yticks([0.5,1.5], labels = ["Taken", "Taken Scrambled"])
    plt.title("Participant " + str(ss + 1))
    plt.show()


LOPOPerfT = pd.concat(subPerf).sort_index(); print(LOPOPerfT.groupby("Model").mean().sort_values('Balanced Accuracy',ascending = False))

results = LOPOPerfT.groupby("Model").mean().sort_index()
results = results.drop(labels = ["Accuracy", "F1 Score"], axis = 1)
results['Balanced Accuracy SD'] = LOPOPerfT.groupby("Model").std().sort_index()["Balanced Accuracy"]
results = results[['Balanced Accuracy','Balanced Accuracy SD','Time Taken']]

# show top 5
fig,ax = plt.subplots(figsize = (15,13))
sns.catplot(x = 'Model',y = 'Balanced Accuracy', \
            data = LOPOPerfT.loc[['GaussianNB','RandomForestClassifier','LogisticRegression','LinearSVC']].reset_index(), \
                estimator = np.mean, kind = 'bar', hue = "Subject")
ax.tick_params(axis='x', which='major', pad=15)
plt.xticks(rotation=45)
plt.show()

sns.set_theme(style = "white", rc={'figure.figsize': (15,10),"xtick.labelsize": 20, \
            'xtick.alignment':'center',"ytick.labelsize": 20,'font.family':'Arial','figure.dpi':500, \
            "axes.labelsize":20,"figure.facecolor":"white",'axes.facecolor':'white',"axes.spines.right": False,"axes.spines.left": False,\
                "axes.spines.top": False,"axes.spines.bottom": False})

def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.02)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)
    # show all..
fig,ax = plt.subplots()
g = sns.barplot(x = 'Model',y = 'Balanced Accuracy', \
            data = LOPOPerfT.reset_index(), \
                estimator = np.mean, palette = sns.cubehelix_palette(n_colors = 27,start = 2.5))
ax.tick_params(axis='x', which='major', pad=15)
plt.ylim([0,1]); plt.axhline(0.5,color = "black")
plt.xticks(rotation=90); plt.xlabel("")
show_values(g)
plt.show()

from sklearn.datasets import make_blobs
import colorcet as cc

blobs, labels = make_blobs(n_samples=1000, centers=25, center_box=(-100, 100))
palette = sns.color_palette(cc.glasbey, n_colors=nSub)


# plot balanced accuracy results for each participant
gAccDf = pd.DataFrame(gAcc.T)
gAccDf = pd.melt(gAccDf,value_vars = np.arange(0,nSub,1))
gAccDf.columns = ["Subject","Balanced Accuracy"]
gAccDf["Subject"] = gAccDf["Subject"] + 1
sns.barplot(data = gAccDf, x = "Subject", y = "Balanced Accuracy", palette = palette)
plt.show() 

sns.set_theme(style = "white", rc={'figure.figsize': (1.56,3),"xtick.labelsize": 12, \
            'xtick.alignment':'center',"ytick.labelsize": 12,'font.family':'Arial','figure.dpi':500, \
            "axes.labelsize":12,"figure.facecolor":"white",'axes.facecolor':'white',"axes.spines.right": False,"axes.spines.left": False,\
                "axes.spines.top": False,"axes.spines.bottom": False})
import matplotlib.transforms as transforms
# same for group level swarmplot
ggAccDf = gAccDf.groupby("Subject").mean().reset_index()
ggAccDf["Subject"] = ggAccDf["Subject"].astype("category") 
fig, ax = plt.subplots()
g = sns.pointplot(data = gAccDf, \
              y = "Balanced Accuracy",hue = "Subject", palette = palette, \
                  x=[""]*len(gAccDf), errorbar=('ci', 95),scale = .5,dodge = True, seed = 59, \
                      ax = ax)
offset = transforms.ScaledTranslation(-5/72., 0, g.figure.dpi_scale_trans)
g.collections[7].set_transform(g.collections[7].get_transform() + offset)
offset = transforms.ScaledTranslation(-5/72., 0, g.figure.dpi_scale_trans)
g.lines[15].set_transform(g.lines[15].get_transform() + offset)
sns.barplot(data = ggAccDf, y = "Balanced Accuracy", palette = "Purples",alpha = 0.5)
plt.ylim([-0.05,1.05])
plt.axhline(0.50, xmin = 0, xmax = 1)
plt.legend([],frameon = False)
plt.show() 


# CIs for balanced accuracy
ba = LOPOPerfT.reset_index();
# get names of all models
modName = ba["Model"].unique().tolist()
ba["CILow"] = 0; ba["CIHigh"] = 0; ba["SE"] = 0;
for name in modName:
    idx = ba["Model"] == name
    temp = ba[idx]["Balanced Accuracy"]
    SE = sem(temp) * 2
    ba.loc[idx.values,"CILow"] = np.mean(temp) - SE 
    ba.loc[idx.values,"CIHigh"] = np.mean(temp) + SE 
    ba.loc[idx.values,"SE"] = SE 

    
ba.reset_index().to_csv("NaciTKNClassifers.csv")

# show distribution for each person and condition
gDist = gProb.reshape((-1,1)); subCol = np.repeat(np.array(range(0,nSub)),gDist.shape[0]/nSub);

condName = ["Taken", "Taken Scrambled"]
nData = np.ones((nSub * nIter * nCond * nCond,5)) * np.nan; counter = 0
for ss in range(0,nSub):
    for ii in range(0,nIter):
        for ncl in condIdx:
            for ncp in condIdx:
                nData[counter,0] = (gProb[ss,ii,ncl-2,ncp-2])
                nData[counter,1] = ss + 1; nData[counter,2] = ii + 1;
                nData[counter,3] = ncl; nData[counter,4] = ncp
                counter += 1

distDF = pd.DataFrame(data = nData, columns = ["Probability","Subject","Iteration","Actual Label","Predicted Label"])
CondMap = {2:"Taken",3:"Taken Scrambled"}

from sklearn.datasets import make_blobs
import colorcet as cc

sns.set_theme(style = "white", rc={'figure.figsize': (3.14,2),"xtick.labelsize": 12, \
            'xtick.alignment':'center',"ytick.labelsize": 12,'font.family':'Arial','figure.dpi':500, \
            "axes.labelsize":12,"figure.facecolor":"white",'axes.facecolor':'white',"axes.spines.right": False,"axes.spines.left": False,\
                "axes.spines.top": False,"axes.spines.bottom": False})
    
blobs, labels = make_blobs(n_samples=1000, centers=25, center_box=(-100, 100))
palette = sns.color_palette(cc.glasbey, n_colors=nSub)

distDF["Actual Label"] = distDF["Actual Label"].map(CondMap)
distDF["Predicted Label"] = distDF["Predicted Label"].map(CondMap)
distDF["Subject"] = distDF["Subject"].astype("int")
distDF[["Subject","Actual Label","Predicted Label"]] = distDF[["Subject","Actual Label","Predicted Label"]].astype("category")


g = sns.FacetGrid(data = distDF,col = "Actual Label", col_wrap = 2)
sns.set_palette(palette, nSub, .75)
g.map_dataframe(sns.pointplot, y = "Probability", x = "Predicted Label",\
             hue = "Subject",linestyles = " ", markers = "x", ci = None,palette = palette)
#g.add_legend()
g.map_dataframe(sns.barplot, y = "Probability", x = "Predicted Label",\
            color = "rebeccapurple",alpha = 0.30)
g.refline(y=0.5)
g.set_axis_labels("");
g.set_xticklabels(rotation=45)
g.set_ylabels("Label Confidence")
plt.show()

# g = sns.catplot(data = distDF, y = "Probability", x = "Predicted Label", col = "Actual Label",\
#             palette = "Purples",row = "Subject", kind = "bar")
# g.refline(y=0.5)
# g.set_ylabels("Label Confidence")

import pingouin as pg

resList = []
# compare against chance
for ss in range(1,nSub+1):
    # for each condition ( no need to two choice schemes)
    print("Participant ",ss)
    for nc in range(0,nCond):
        tdat = distDF[distDF["Subject"] == ss]
        tdat = tdat[tdat["Actual Label"] == condName[nc]]
        print("Actual Condition ", condName[nc])
        # for each option within condition
        for ncp in range(0,nCond):
            print("Predicted Condition ", condName[ncp])
            pdat = tdat[tdat["Predicted Label"] == condName[ncp]]["Probability"]
            res = pg.ttest(pdat,.50,alternative = "greater")
            res["subject"] = ss; res["Actual Label"] = condName[nc]; res["Predicted Label"] = condName[ncp]
            resList.append(res)
            print(res)

Results = pd.concat(resList)

# how many subjects are significantly identified in each condition
Intact = Results[Results["Actual Label"] == "Taken"]
print(Intact[Intact["Predicted Label"] == "Taken"]["p-val"])
print(fdrcorrection(Intact[Intact["Predicted Label"] == "Taken"]["p-val"]))
    
Scrambled= Results[Results["Actual Label"] == "Taken Scrambled"]
print(Scrambled[Scrambled["Predicted Label"] == "Taken Scrambled"]["p-val"])
print(fdrcorrection(Scrambled[Scrambled["Predicted Label"] == "Taken Scrambled"]["p-val"]))

# same for group


resList = []
# compare against chance
for nc in range(0,nCond):
    tdat = distDF[distDF["Actual Label"] == condName[nc]]
    print("Actual Condition ", condName[nc])
    # for each option within condition
    for ncp in range(0,nCond):
        print("Predicted Condition ", condName[ncp])
        pdat = tdat[tdat["Predicted Label"] == condName[ncp]]["Probability"]
        res = pg.ttest(pdat,.50,alternative = "greater")
        res["Actual Label"] = condName[nc]; res["Predicted Label"] = condName[ncp]
        resList.append(res)
        print(res)
        
# overall balanced accuracy
bRes = np.concatenate(\
               [fdrcorrection(Intact[Intact["Predicted Label"] == "Taken"]["p-val"])[0],\
                   ~fdrcorrection(Scrambled[Scrambled["Predicted Label"] == "Taken Scrambled"]["p-val"])[0]])
Ycomp = np.array(np.concatenate([np.ones(nSub),np.zeros(nSub)]),dtype = "bool")
balanced_accuracy_score(Ycomp,bRes)