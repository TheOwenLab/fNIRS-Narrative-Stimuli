# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:56:34 2022

@author: Matthew
"""
import pandas as pd
from scipy.io import loadmat
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
# load in information
path = r"D:\Data\NaciRep\HealthyFinalAnalysis"


fpd = os.path.join(path,"tSpaceInd.mat")
fpt = os.path.join(path,"ttestSpaceInd.mat")
fpc = os.path.join(path,"corSpaceInd.mat")

comp = "BYD"

if comp == "BYD":
    task = 0
else:
    task = 2

d = loadmat(fpd)['simMatMask']
t = loadmat(fpt)['ttestMatMask']
c = loadmat(fpc)['corMatMask']


sns.set_theme(style = "white", rc={'figure.figsize': (7.28,5),"xtick.labelsize": 12, \
            'xtick.alignment':'center',"ytick.labelsize": 12,'font.family':'Arial','figure.dpi':500, \
            "axes.labelsize":12,"figure.facecolor":"white",'axes.facecolor':'white',"axes.spines.right": False,"axes.spines.left": False,\
                "axes.spines.top": False,"axes.spines.bottom": False})
    
# exp parameters
nSub = d.shape[0]; nCond = d.shape[1];
data = np.ones((nSub * (nSub - 1) * nCond,6)) * np.nan
CondNames = ["BYD","BYD Scrambled","Taken","Taken Scrambled"]
nSubList = np.arange(0,nSub,1)
# make into dataframes
# unpack data
counter = 0
for ss in range(0,nSub):
    for nc in range(0,nCond):
        for ls in range(0,nSub-1):
           data[counter,0] = d[ss,nc,ls]
           data[counter,1] = t[ss,nc,ls]
           data[counter,2] = c[ss,nc,ls]
           # which subjects are we comparing?
           cSub = nSubList[nSubList != ss]
           data[counter,3] = ss + 1
           data[counter,4] = cSub[ls] + 1
           data[counter,5] = nc
           counter += 1

# create columns for healthy distance values
cNames = ["Dist","T-Score","Correlation","Subject","SubjectComp","Condition"]
# create df
df = pd.DataFrame(data,columns = cNames)
df["Condition"] = df["Condition"].map({0:CondNames[0],1:CondNames[1],2:CondNames[2],\
                                       3:CondNames[3]})
df["Subject"] = df["Subject"].astype("int")
df["SubjectComp"] = df["SubjectComp"].astype("int")

from sklearn.datasets import make_blobs
import colorcet as cc

blobs, labels = make_blobs(n_samples=1000, centers=25, center_box=(-100, 100))
palette = sns.color_palette(cc.glasbey, n_colors=nSub)


# plot results
# first across group
sns.barplot(data = df, x = "Condition", y = "Dist")
plt.show()
sns.barplot(data = df, x = "Condition", y = "Correlation")
plt.show()
sns.barplot(data = df, x = "Condition", y = "T-Score")
plt.show()


# next across subjects
sns.catplot(data = df, x = "SubjectComp", y = "Dist")
plt.show()
sns.catplot(data = df, x = "Condition", y = "Dist")
plt.show()
sns.boxplot(data = df, x = "Condition", y = "Dist",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()


sns.catplot(data = df, x = "SubjectComp", y = "T-Score")
plt.show()
sns.catplot(data = df, x = "Condition", y = "T-Score")
plt.show()
sns.boxplot(data = df, x = "Condition", y = "T-Score",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

sns.catplot(data = df, x = "SubjectComp", y = "Correlation")
plt.show()
sns.catplot(data = df, x = "Condition", y = "Correlation")
plt.show()
sns.pointplot(data = df, x = "Condition", y = "Correlation",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

# a bit complicated to follow -- just show for the Intact Conditions?
dfB = df[(df["Condition"] == "BYD")]; dfT = df[(df["Condition"] == "Taken")];

sns.swarmplot(data = dfB, x = "SubjectComp", y = "Correlation")
sns.pointplot(data = dfB, x = "SubjectComp", y = "Correlation",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

sns.swarmplot(data = dfB, x = "SubjectComp", y = "Dist")
sns.pointplot(data = dfB, x = "SubjectComp", y = "Dist",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

sns.swarmplot(data = dfB, x = "SubjectComp", y = "T-Score")
sns.pointplot(data = dfB, x = "SubjectComp", y = "T-Score",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

# plot Taken

sns.swarmplot(data = dfT, x = "SubjectComp", y = "Correlation",hue = "Subject",palette = palette)
sns.pointplot(data = dfT, x = "SubjectComp", y = "Correlation",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

sns.swarmplot(data = dfT, x = "SubjectComp", y = "Dist",hue = "Subject",palette = palette)
sns.pointplot(data = dfT, x = "SubjectComp", y = "Dist",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

sns.swarmplot(data = dfT, x = "SubjectComp", y = "T-Score",hue = "Subject",palette = palette)
sns.pointplot(data = dfT, x = "SubjectComp", y = "T-Score",hue = "SubjectComp",palette = palette)
plt.legend([],frameon = False)
plt.show()

# plot all the continuous variables together
sns.pairplot(data = dfB.loc[:,["Dist","T-Score","Correlation"]])
plt.show()
sns.pairplot(data = dfT.loc[:,["Dist","T-Score","Correlation"]])
plt.show()

# it seems that correlation and dissimilarity are related -- but suprisingly t-score is not?

# save df
df.to_csv("SimilarityInd.csv")

# run some statistical analyses
pg.pairwise_tukey(data=df, dv='Dist', between='Subject')
pg.pairwise_tukey(data=df, dv='Correlation', between='Subject')
pg.pairwise_tukey(data=df, dv='T-Score', between='Subject')

# compare each healthy control's dissimilarity
tResDis = [];
hdf = dfB.copy()
for ss in range(1,nSub + 1):
    nDF = hdf.copy(); 
    nDF["Comp"] = "Group"
    nDF["Comp"].loc[hdf["SubjectComp"] == ss] = "LO"
    res = pg.welch_anova(data = nDF, dv = "Dist", between = "Comp")
    

    print("Subject Number ",ss)
    print(res)
    print(pg.ttest(x = nDF[nDF["Comp"] == "LO"]["Dist"].values, \
                   y = nDF[nDF["Comp"] == "Group"]["Dist"].values, \
                       alternative = "greater", correction = True))
    tres = pg.ttest(x = nDF[nDF["Comp"] == "LO"]["Dist"].values, \
                   y = nDF[nDF["Comp"] == "Group"]["Dist"].values, \
                       alternative = "greater", correction = True)
    tResDis.append(tres)


from statsmodels.stats.multitest import fdrcorrection

# concat p-vals
trDF = pd.concat(tResDis)
p_val = trDF['p-val'].values
# fdr correct results
(h,q) = fdrcorrection(p_val)
print(h,q)

# compare each healthy control's t-score
tResT = [];
hdf = dfB.copy()
for ss in range(1,nSub + 1):
    nDF = hdf.copy(); 
    nDF["Comp"] = "Group"
    nDF["Comp"].loc[hdf["SubjectComp"] == ss] = "LO"
    res = pg.welch_anova(data = nDF, dv = "T-Score", between = "Comp")
    

    print("Subject Number ",ss)
    print(res)
    print(pg.ttest(x = nDF[nDF["Comp"] == "LO"]["T-Score"].values, \
                   y = nDF[nDF["Comp"] == "Group"]["T-Score"].values, \
                       alternative = "greater", correction = True))
    tres = pg.ttest(x = nDF[nDF["Comp"] == "LO"]["T-Score"].values, \
                   y = nDF[nDF["Comp"] == "Group"]["T-Score"].values, \
                       alternative = "greater", correction = True)
    tResT.append(res)



# concat p-vals
trDF = pd.concat(tResT)
p_val = trDF['p-unc'].values
# fdr correct results
(h,q) = fdrcorrection(p_val)
print(h,q)


# next compare the correlation values


from sklearn.datasets import make_blobs
import colorcet as cc

# get confidence interval of dist scores
mTOS = df[df["Condition"] == "Taken"]["Dist"].mean()
sTOS = df[df["Condition"] == "Taken"]["Dist"].std()
ciTOS = sTOS/np.sqrt(nSub) * 1.96

print("Average Dist",np.round(mTOS,2))
print("Average Dist SD",np.round(sTOS,2))
print("CI",mTOS - ciTOS  ,"-", mTOS + ciTOS)
print("TOS Range",df[df["Condition"] == "BYD"]["Dist"].min(),"-",\
      df[df["Condition"] == "BYD"]["Dist"].max())

# is taken different from byd
pg.ttest(x = df[df["Condition"] == "BYD"]["Dist"].values, \
                   y = df[df["Condition"] == "Taken"]["Dist"].values\
                       , correction = "auto")   

