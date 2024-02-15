
# inport helper functions
import pandas as pd
import os
import numpy as np
import seaborn as sns
import numpy.matlib
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
import pingouin as pg
from scipy.io import loadmat
from scipy.stats import sem
from joblib import dump, load

# import machine learning helper functions and preprocessing
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score,balanced_accuracy_score,confusion_matrix,recall_score,precision_score,f1_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from hyperopt import fmin,tpe,hp,Trials
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA

# import classifiers 
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,\
    VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


# set experiment variables
nSub = 26; nCond = 2; nChan = 121; nHb = 2; SSList = [7, 28, 51, 65, 74, 91, 111, 124]
condIdx = [0,1]

# change directory
path = r'D:\Data\NaciRep\Publication\SensitivityAnalysis'
os.chdir(path)

# load in original and hold out ISCs
file = "ISC.mat"; fileh = "ISCHoldOut.mat"

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

# combine all relavent info
df["ChannelName"] = cNameArr.tolist() * nSub * nCond
df["Subject"] = df["Subject"] + 1

# set up np array for ML (drop condition info)
Xs = X[:,1:]; Y = X[:,0]

# uppack hold out ISCs
# for each subject
dataX = []
for ss in range(0,nSub):
    td = ISCML[:,:,ss,:]
    # and for each condition
    tdc = np.concatenate([td[0,:,:],td[1,:,:]],axis = 1)    
    iscTemp = tdc
    dataX.append(iscTemp)

xML = np.stack(dataX); yML = np.repeat([0,1],repeats = (nSub - 1))

# set Ys to 0 and 1
Y = Y == 0; yML = yML == 0;

CondList = set([2,3])

# set up loop variables
nSubList = np.array(range(0,nSub-1))
np.random.seed(59)
nFold = 3; 

np.random.RandomState(23)

# initalize variables in memory
groupPerf = []; subPerf = [];
gAcc = np.ones((nSub,nChan))


param_choices = {'penalty':["l1","l2"],\
                  'solver':['liblinear','saga'],\
                      'max_iter':[100,1000],\
                          'n_estimators': [50, 100, 1000],\
                          'max_depth':[None] + list(np.arange(1, 4)),\
                          'algorithm': ['SAMME', 'SAMME.R'],\
                              'criterion':['gini', 'entropy'],\
                               'max_features':[None, 'sqrt', 'log2'],
                               'bootstrap': [True, False],
                                   'bootstrap_features': [True, False],
                                   'oob_score': [True, False],
                                       'fit_intercept':[True, False],\
                                               'n_neighbors':list(range(2, 4)),\
                                                   'metric':['euclidean', 'manhattan'],\
                                                       'p':[1,2],\
                                                           'weights':['uniform', 'distance'],\
                                                               'kernel':['rbf','linear'],\
                           'shrinkage':[None, 'auto'] + list(np.linspace(0, 1, 10)),\
                                   'shrinking':[True,False],\
                                       'objective':['binary:logistic','binary:hinge','rank:map'],\
                                           'booster':['gbtree'],\
                                               # PCA 
                                                       'n_components':[1,2,5,10],\
                                                                   'max_iter_predict': list(range(50, 501, 50)),  # Maximum iterations for the prediction
                                                                   'n_restarts_optimizer': list(range(0, 11)),\
                                                                       'rbf_length_scale': list(np.logspace(-2, 2, 10))}  # Length scale for RBF kernel}  # Number of restarts for the optimizer

linear_space = {
    'C': hp.loguniform('C', np.log(1e-2), np.log(1e2)),
    'penalty': hp.choice('penalty', param_choices['penalty']),
    'solver': hp.choice('solver', param_choices['solver']),
    'max_iter': hp.choice('max_iter', param_choices['max_iter'])}  

svm_space = {
    'C': hp.loguniform('C', np.log(1e-2), np.log(1e2)),
    'kernel': hp.choice('kernel',param_choices['kernel'])  
}


centroid_space = {
    'metric': hp.choice('metric', param_choices['metric']),
    'shrink_threshold': hp.uniform('shrink_threshold', 0, 1)  # Depending on the scale of your data, adjust the range accordingly
}


knn_space = {
    'n_neighbors': hp.choice('n_neighbors',param_choices['n_neighbors']),
    'weights': hp.choice('weights', param_choices['weights']),
    'p': hp.choice('p', param_choices['p'])  # limit to L1 and L2 distances
}

gnb_space = {
    'var_smoothing': hp.loguniform('var_smoothing', np.log(1e-2), np.log(1e2))
}

bnb_space = {
    'alpha': hp.uniform('alpha', 0, 2),
    'binarize': hp.uniform('binarize', 0.0, 1.0),
    'fit_prior': hp.choice('fit_prior', [True, False])
}

ada_space = {
    'n_estimators': hp.choice('n_estimators', param_choices['n_estimators']),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-1), np.log(1e2)), 
    'algorithm': hp.choice('algorithm', param_choices['algorithm']),
}
   
etc_space = {
    'n_estimators': hp.choice('n_estimators', param_choices['n_estimators']),
    'criterion': hp.choice('criterion', param_choices['criterion']),
    'max_depth': hp.choice('max_depth', param_choices['max_depth']),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 1),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
    'max_features': hp.choice('max_features', param_choices['max_features']),
    'bootstrap': hp.choice('bootstrap', param_choices['bootstrap']),
    'class_weight': hp.choice('class_weight', param_choices['class_weight'])
}

bg_space = {
    'n_estimators': hp.choice('n_estimators', param_choices['n_estimators']),
    'bootstrap': hp.pchoice('bootstrap', [(1,param_choices['bootstrap'][0]),\
                                          (0,param_choices['bootstrap'][1])]),
    'bootstrap_features': hp.choice('bootstrap_features', param_choices['bootstrap_features']),
    'oob_score': hp.choice('oob_score', param_choices['oob_score'])
}
    

ridge_space = {
    'alpha': hp.loguniform('alpha', np.log(1e-2), np.log(1e2)),  # Regularization strength
    'fit_intercept': hp.choice('fit_intercept', param_choices['fit_intercept']),
    'max_iter': hp.choice('max_iter', param_choices['max_iter']),
    'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-3))
}
 
labelp_space = {
    'gamma': hp.loguniform('gamma', np.log(1e-2), np.log(1e2)),  # Kernel coefficient for rbf
    'max_iter': hp.choice('max_iter', param_choices['max_iter']),
    'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-3))
}

labels_space = {
    'gamma': hp.loguniform('gamma', np.log(1e-2), np.log(1e2)),  # Kernel coefficient for rbf
    'alpha': hp.uniform('alpha', 0.01, 0.99),  # Clamping factor
    'max_iter': hp.choice('max_iter', param_choices['max_iter']),
    'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-3))
}

lda_space = None



qda_space = {
    'reg_param': hp.uniform('reg_param', 0, 1)  # Regularization parameter
}


nu_space = {
    'nu': hp.uniform('nu', 0.1, 0.9),  # An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors
    'kernel': hp.choice('kernel', param_choices['kernel']),  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    'shrinking': hp.choice('shrinking', param_choices['shrinking']),
    'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-3))  # Tolerance for stopping criterion.
}

xgb_space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-2), np.log(1e2)),  # Controls the contribution of each tree in the final outcome.
    'n_estimators': hp.choice('n_estimators', param_choices['n_estimators']),  # Number of boosting rounds.
    'max_depth': hp.choice('max_depth', param_choices['max_depth']),  # Depth of the tree.
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),  # Minimum sum of instance weight needed in a child.
    'gamma': hp.loguniform('gamma', np.log(1e-2), np.log(1e2)),  # Minimum loss reduction required to make a further partition.
    'subsample': hp.quniform('subsample', 0.75, 1, 0.05),  # Proportion of training data to randomly sample in each boosting round.
    'colsample_bytree': hp.quniform('colsample_bytree', 0.75, 1, 0.05),  # Proportion of features to randomly sample for building each tree.
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.75, 1, 0.05),  # Proportion of features to randomly sample for building each level.
    'colsample_bynode': hp.quniform('colsample_bynode', 0.75, 1, 0.05),  # Proportion of features to randomly sample for building each node.
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-2), np.log(1e2)),  # L1 regularization term on weights.
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-2), np.log(1e2)),  # L2 regularization term on weights.
    'objective': hp.choice('objective', param_choices['objective']),  # Specifies the learning task.

}



gp_space = {
    'max_iter_predict': hp.choice('max_iter_predict', param_choices['max_iter_predict']),
    'n_restarts_optimizer': hp.choice('n_restarts_optimizer', param_choices['n_restarts_optimizer']),
    'kernel': hp.choice('kernel', [RBF(length_scale) for length_scale in param_choices['rbf_length_scale']])
    
}


pca_opt = {'n_components':hp.choice('n_components',param_choices['n_components'])}



# initalize estimators 
ESTIMATORS = [("et", ExtraTreesClassifier(random_state = 23,class_weight = "balanced"),[etc_space]),\
              ("ab", AdaBoostClassifier(), [ada_space]), \
               ("bc", BaggingClassifier(),[bg_space]),\
                   ("gpc",GaussianProcessClassifier(random_state=23),[gp_space]),\
              ("lr", LogisticRegression(random_state = 23,class_weight = "balanced"),[linear_space]),\
                  ("rc", RidgeClassifier(random_state = 23,class_weight = "balanced"),[ridge_space]),\
                  ("bnb", BernoulliNB(),[bnb_space]), 
                  ("gnb", GaussianNB(),[gnb_space]),\
                  ("knn", KNeighborsClassifier(),[knn_space]),\
                      ("nc", NearestCentroid(),[centroid_space]),\
                      ("lp",LabelPropagation(),[labelp_space]),\
                          ("ls",LabelSpreading(),[labels_space]),\
                          ("lda",LinearDiscriminantAnalysis(),[lda_space]), \
                              ("qda",QuadraticDiscriminantAnalysis(),[qda_space]), \
                                  ("svc",SVC(random_state = 23,class_weight = "balanced"),[svm_space]), \
                                      ("nsvc",NuSVC(random_state = 23),[nu_space]),\
                                          ("xgb",XGBClassifier(random_state = 23),[xgb_space]),\
                                              ]




def evaluate_model_general(params,X,y,model,i):

    # Instantiate your model with the given hyperparameters
    params = params[0]
    pipe_xgb = Pipeline([('scaler',MinMaxScaler()),\
                         ('model',model.set_params(**params))])   
    
    # Perform cross-validation and calculate scores
    balanced_accuracy = cross_val_score(pipe_xgb, X, y, \
                                        cv = StratifiedKFold(n_splits=nFold), \
                             scoring = "balanced_accuracy")

    #print(balanced_accuracy)
# Return the average score to be minimized/maximized by hyperopt
    return -np.median(balanced_accuracy)  # Negative sign for minimization



def evaluate_model_pca(params,X,y,model,i):

    # Instantiate your model with the given hyperparameters
    hyp_params = params[0]; pca_params = params[1]
    pipe_pca = Pipeline([('scaler',StandardScaler(with_std=False)),\
                         ('pca',PCA(n_components = pca_params['n_components'],random_state = 23)),\
                         ('model',model.set_params(**hyp_params))])   
    
    # Perform cross-validation and calculate scores
    balanced_accuracy = cross_val_score(pipe_pca, X, y, \
                                        cv = StratifiedKFold(n_splits=nFold), \
                             scoring = "balanced_accuracy")

    #print(balanced_accuracy)
# Return the average score to be minimized/maximized by hyperopt
    return -np.median(balanced_accuracy)  # Negative sign for minimization

def translate_hyperparameters(best_results, param_choices):
    actual_values = {}
    for param, value in best_results.items():
        if param in param_choices:
            actual_values[param] = param_choices[param][value]
        else:
            actual_values[param] = value
    return actual_values



# initialize variables to store performance 
BEST = []; MODELS = []; MODEL_PERFORMANCE = []; SUBPERFORMANCE = []; ACTUAL = np.zeros((len(ESTIMATORS),nSub,nCond))
PREDICTED = np.zeros((len(ESTIMATORS),nSub,nCond));  MODEL_OUTPUT = np.zeros((len(ESTIMATORS),nSub,nCond))
VALIDATION_PERFORMANCE = []; 
GROUPPREDICTION = np.zeros((nSub,nCond,nChan)); NCOMPONENTS = []; 
RECALL = np.zeros((len(ESTIMATORS),nSub)); PRECISION = np.zeros((len(ESTIMATORS),nSub))
PREDICTPROBA = np.ones((len(ESTIMATORS),nSub,2)) * 1
VOTE = [];

randcounter = 0;
for cc in range(0,nChan):
    for ss in range(0,nSub):
        # get subject data and set as train set
        print(ss + 1)
        SUBMODELS = []; SUBPERFORMANCE = []; SUBVALPERFORMANCE = []
        # get train and test sets
        X_test = Xs[Xs[:,0] == ss,1:]; y_test = Y[Xs[:,0] == ss]
        X_train = xML[ss,:,:].T; y_train = yML
        
        # take only areas within the mask
        X_train = X_train[:,cc].reshape(-1,1)
        X_test = X_test[:,cc].reshape(-1,1)
    
        # randomize the arrays
        trainrand = np.random.permutation(y_train.shape[0]); testrand = np.random.permutation(y_test.shape[0]);
        X_test = X_test[testrand,:]; y_test = y_test[testrand];
        X_train = X_train[trainrand,:]; y_train = y_train[trainrand]
    
        
        # start optimizating each classifier
        for idx,estimator in enumerate(ESTIMATORS):
            model_name = estimator[0]
            params = estimator[-1] # at the end of the tuple
            model = estimator[1] # in the middle
            randcounter += 1
            if params[0] is not None:
                if len(params) > 1: # if PCA
                    trials = Trials()
                    best = fmin(fn=lambda params: evaluate_model_pca(params,X_train,y_train,model,randcounter),\
                                space=params, \
                                algo=tpe.suggest, max_evals=5 * len(params[0]), trials=trials,verbose = True,\
                                    rstate = np.random.default_rng(23))
                                
                    
                    # extract the best trial
                    best_values = translate_hyperparameters(best, param_choices)
            
                    
                    # store the hyperparameters
                    BEST.append(best_values)
                    
                    # store validation score
                    SUBVALPERFORMANCE.append(trials.best_trial['result']['loss'] * -1)
                    
                    
                    pca_key = ['n_components']
                    n_components = [best_values.pop(key) for key in pca_key][0]
                    NCOMPONENTS.append(n_components)
                    
                    # store fit model
                    tmodel =  Pipeline([('scaler',StandardScaler(with_std=False)),\
                                         ('pca',PCA(n_components = n_components,random_state = 23)),\
                                         ('model',model.set_params(**best_values))])   
                    tmodel.fit(X_train,y_train)
                    print(confusion_matrix(y_train,tmodel.predict(X_train)))
                    RECALL[idx,ss] = recall_score(y_train,tmodel.predict(X_train))
                    PRECISION[idx,ss] = precision_score(y_train,tmodel.predict(X_train))
                    SUBMODELS.append((model_name,tmodel))
                    
                    # predict test set with current model 
                    SUBPERFORMANCE.append(balanced_accuracy_score(y_test,tmodel.predict(X_test)))
                    ACTUAL[idx,ss,:] = [y_test[np.where(y_test)[0]][0],y_test[np.where(~y_test)[0]][0]]
                    res = tmodel.predict(X_test)
                    PREDICTED[idx,ss,:] = [res[np.where(y_test)[0]][0],res[np.where(~y_test)[0]][0]]
                    
                    # if model has predict proba, store that as well
                    if hasattr(tmodel, 'predict_proba'):
                        probs = tmodel.predict_proba(X_test)
                        print(probs)
                        print(model_name)
                    
                        PREDICTPROBA[idx,ss,0] = probs[np.where(y_test)[0]][0][1]
                        PREDICTPROBA[idx,ss,1] = probs[np.where(~y_test)[0]][0][0]
    
        
                    
                else: # if not PCA
                    trials = Trials()
                    best = fmin(fn=lambda params: evaluate_model_general(params,X_train,y_train,model,randcounter),\
                                space=params, \
                                algo=tpe.suggest, max_evals= 5 * len(params[0]), trials=trials,verbose = True,\
                                    rstate = np.random.default_rng(23))
                    
                        
                     # kernels break the translate hyperparameter function
                    if model_name == 'gpc':
                        kernel_key = 'kernel'
                        length_scale = param_choices['rbf_length_scale'][best['kernel']]
                        best.pop(kernel_key)
                        # extract the best trial
                        best_values = translate_hyperparameters(best, param_choices)
                        # add back custom kernel
                        best_values['kernel'] = RBF(length_scale)
                        
                        # store the hyperparameters
                        BEST.append(best_values)
                        
                        # store validation score
                        SUBVALPERFORMANCE.append(trials.best_trial['result']['loss'] * -1)
                        
                        
                        # store fit model
                        tmodel =  Pipeline([('scaler',MinMaxScaler()),\
                                             ('model',model.set_params(**best_values))])   
                        tmodel.fit(X_train,y_train)
                        print(confusion_matrix(y_train,tmodel.predict(X_train)))
                        RECALL[idx,ss] = recall_score(y_train,tmodel.predict(X_train))
                        PRECISION[idx,ss] = precision_score(y_train,tmodel.predict(X_train))
                        SUBMODELS.append((model_name,tmodel))
                        
                        # predict test set with current model 
                        SUBPERFORMANCE.append(balanced_accuracy_score(y_test,tmodel.predict(X_test)))
                        ACTUAL[idx,ss,:] = [y_test[np.where(y_test)[0]][0],y_test[np.where(~y_test)[0]][0]]
                        res = tmodel.predict(X_test)
                        PREDICTED[idx,ss,:] = [res[np.where(y_test)[0]][0],res[np.where(~y_test)[0]][0]]
                        
                        # if model has predict proba, store that as well
                        if hasattr(tmodel, 'predict_proba'):
                            probs = tmodel.predict_proba(X_test)
                            print(probs)
                            print(model_name)
                        
                            PREDICTPROBA[idx,ss,0] = probs[np.where(y_test)[0]][0][1]
                            PREDICTPROBA[idx,ss,1] = probs[np.where(~y_test)[0]][0][0]
    
                    else:
                        # extract the best trial
                        best_values = translate_hyperparameters(best, param_choices)
                        
                        # store the hyperparameters
                        BEST.append(best_values)
                        
                        # store validation score
                        SUBVALPERFORMANCE.append(trials.best_trial['result']['loss'] * -1)
                        
                        # store fit model
                        tmodel = make_pipeline(MinMaxScaler(),\
                                         model.set_params(**best_values))
                        tmodel.fit(X_train,y_train)
                        print(confusion_matrix(y_train,tmodel.predict(X_train)))
                        RECALL[idx,ss] = recall_score(y_train,tmodel.predict(X_train))
                        PRECISION[idx,ss] = precision_score(y_train,tmodel.predict(X_train))
                        SUBMODELS.append((model_name,tmodel))
                        
                        # predict test set with current model 
                        SUBPERFORMANCE.append(balanced_accuracy_score(y_test,tmodel.predict(X_test)))
                        ACTUAL[idx,ss,:] = [y_test[np.where(y_test)[0]][0],y_test[np.where(~y_test)[0]][0]]
                        res = tmodel.predict(X_test)
                        PREDICTED[idx,ss,:] = [res[np.where(y_test)[0]][0],res[np.where(~y_test)[0]][0]]
                        
                        # if model has predict proba, store that as well
                        if hasattr(tmodel, 'predict_proba'):
                            probs = tmodel.predict_proba(X_test)
                            print(probs)
                            print(model_name)
                            
                            PREDICTPROBA[idx,ss,0] = probs[np.where(y_test)[0]][0][1]
                            PREDICTPROBA[idx,ss,1] = probs[np.where(~y_test)[0]][0][0]
    
        
                
            else:
                # store the hyperparameters
                BEST.append(None)
                
                # store fit model
                tmodel = make_pipeline(MinMaxScaler(),\
                                 model)
                tmodel.fit(X_train,y_train)
                SUBVALPERFORMANCE.append(balanced_accuracy_score(y_train,tmodel.predict(X_train)))
                SUBMODELS.append((model_name,tmodel))
                RECALL[idx,ss] = recall_score(y_train,tmodel.predict(X_train))
                PRECISION[idx,ss] = precision_score(y_train,tmodel.predict(X_train))
                # predict test set with current model 
                SUBPERFORMANCE.append(balanced_accuracy_score(y_test,tmodel.predict(X_test)))
                ACTUAL[idx,ss,:] = [y_test[np.where(y_test)[0]][0],y_test[np.where(~y_test)[0]][0]]
                res = tmodel.predict(X_test)
                PREDICTED[idx,ss,:] = [res[np.where(y_test)[0]][0],res[np.where(~y_test)[0]][0]]
        
                # if model has predict proba, store that as well
                if hasattr(tmodel, 'predict_proba'):
                    probs = tmodel.predict_proba(X_test)
                    print(probs)
                    print(model_name)
            
                    PREDICTPROBA[idx,ss,0] = probs[np.where(y_test)[0]][0][1]
                    PREDICTPROBA[idx,ss,1] = probs[np.where(~y_test)[0]][0][0]
    
                
        
        MODELS.append(SUBMODELS)
        MODEL_PERFORMANCE.append(SUBPERFORMANCE) 
        VALIDATION_PERFORMANCE.append(SUBVALPERFORMANCE)
        # create a pipeline 
        stk = VotingClassifier(estimators = SUBMODELS, voting = "hard", weights = 1+np.array(SUBVALPERFORMANCE)) # make own voting classifier
       # stk = StackingClassifier(estimators = SUBMODELS,\
       #                        final_estimator = LinearSVC(penalty = "l2",dual = "auto",random_state=23), \
        #                       #final_estimator = XGBClassifier(random_state=23), \
        #                       #cv = "prefit")
        #                        cv = StratifiedKFold(n_splits=nFold,random_state=23,\
        #                                            shuffle=True),passthrough = True)
        
    
        pipe = Pipeline([#('scaler', MinMaxScaler()),\
                         ('stk', stk)])
        
        pipe.fit(X_train, y_train)
        
        VOTE.append(pipe)
        
        accPred  = pipe.predict(X_test) 
        GROUPPREDICTION[ss,:,cc] = [accPred[np.where(y_test)[0]][0],accPred[np.where(~y_test)[0]][0]]
        
        
        # check if overfitting
        print(balanced_accuracy_score(y_train,pipe.predict(X_train)))  
        print(balanced_accuracy_score(y_test,accPred))
        gAcc[ss,cc] = balanced_accuracy_score(y_test,accPred) 




# a predictive channel is one that is accurate across particpants
# mean accuracy for each channel
pAcc = np.mean(gAcc,axis = 0)
# and is confident across participants


# save both confidence and mask
from scipy.io import savemat
savemat("BYDSingleChannelResVote.mat",{"pAcc":pAcc})