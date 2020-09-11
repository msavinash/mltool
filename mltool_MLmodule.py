from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import  KNeighborsRegressor

from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, f1_score,precision_score,recall_score,confusion_matrix

from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import pandas as pd
import json
import os
import pickle

USERS_DIR = "USERS/"
DATASET_UPLOAD_FOLDER = '/datasets'

def getCurrentSession(current_user):
    if not os.path.exists(USERS_DIR+current_user+'/session.pkl'):
        session = {}
        return session
    else:
        pfile = open(USERS_DIR+current_user+"/session.pkl","rb")
        session = pickle.load(pfile)
        return session

def readFile(current_user):
    session = getCurrentSession(current_user)
    filename = session['filename']
    df = pd.read_csv(USERS_DIR+current_user+DATASET_UPLOAD_FOLDER+'/'+filename)
    return df

def trainModel(df,modelName, target,cols,model,test_s=0.3,reg=False):
    X = df[cols]  # Features
    y = df[target]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s, random_state=1)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    data=modelCharacteristics(modelName, y_test,y_pred,reg=False)
    return data, model


def modelCharacteristics(modelName, y_test,y_pred, reg):
    data = {}
    if reg:
        mse = mean_squared_error(y_test, y_pred)
        data["mse"] = mse
    else:
        accuracy = round(accuracy_score(y_test, y_pred), 6)
        f1 = round(f1_score(y_test, y_pred), 6)
        pr = round(precision_score(y_test, y_pred), 6)
        r = round(recall_score(y_test, y_pred), 6)

        cm = confusion_matrix(y_test, y_pred)
        cm=cm.tolist()

        data = {"Model": modelName, "Accuracy": accuracy, "f1-score": f1, "Precision": pr, "Recall": r, "Confusion Matrix": cm}
    return data


def setupModel(session, current_user):
    df = readFile(current_user)
    targetColumn=session['targetColumn']
    cols=session['columns']
    data={}
    model=""
    if "all" in cols:
        cols = list(df.columns)
    if(session['model']=="Decision Tree"):
        criterionV = session['modelArguments']['criterion']
        splitterV = session['modelArguments']['splitter']
        max_depthV = session['modelArguments']['max_depth']
        if(max_depthV == 'None'):
            max_depthV = None
        elif(max_depthV != 'None'):
            max_depthV=int(max_depthV)
        if(session['modelArguments']['min_samples_split']>1):
            session['modelArguments']['min_samples_split']=int(session['modelArguments']['min_samples_split'])
        elif(session['modelArguments']['min_samples_split']<=1):
            session['modelArguments']['min_samples_split']=float(session['modelArguments']['min_samples_split'])
        min_samples_splitV = session['modelArguments']['min_samples_split']
        min_samples_leafV = int(session['modelArguments']['min_samples_leaf'])
        max_leaf_nodesV = session['modelArguments']['max_leaf_nodes']
        if(max_leaf_nodesV == 'None'):
           max_leaf_nodesV = None
        elif(max_leaf_nodesV != 'None'):
            max_leaf_nodesV=int(max_leaf_nodesV)
        model = DecisionTreeClassifier(criterion=criterionV, splitter=splitterV, max_depth=max_depthV, min_samples_split=min_samples_splitV, min_samples_leaf=min_samples_leafV, max_leaf_nodes=max_leaf_nodesV)
    elif(session['model']=="SVM"):
        CV = float(session['modelArguments']['C'])
        kernelV = session['modelArguments']['kernel']
        degreeV = int(session['modelArguments']['degree'])
        gammaV = session['modelArguments']['gamma']
        coef0V = float(session['modelArguments']['coef0'])
        shrinkingV = session['modelArguments']['shrinking']
        if(shrinkingV == 'True'):
            shrinkingV = True
        elif(shrinkingV == 'False'):
            shrinkingV = False
        probabilityV = session['modelArguments']['probability']
        if(probabilityV == 'True'):
            probabilityV = True
        elif(probabilityV == 'False'):
            probabilityV = False
        tolV = float(session['modelArguments']['tol'])
        decision_function_shapeV = session['modelArguments']['decision_function_shape']
        model=svm.SVC(C=CV, kernel=kernelV, degree=degreeV, gamma=gammaV, coef0=coef0V, shrinking=shrinkingV, probability=probabilityV, tol=tolV, decision_function_shape=decision_function_shapeV)
    elif(session['model']=="Neural Network"):
        solverV = session['modelArguments']['solver']
        alphaV = float(session['modelArguments']['alpha'])
        hidden_layer_sizesV = session['modelArguments']['hidden_layer_sizes']
        l = hidden_layer_sizesV.split(',')
        hidden_layer_sizesV = tuple([int(i) for i in l])
        activationV = session['modelArguments']['activation']
        batch_sizeV = int(session['modelArguments']['batch_size'])
        learning_rateV = session['modelArguments']['learning_rate']
        learning_rate_initV = float(session['modelArguments']['learning_rate_init'])
        power_tV = float(session['modelArguments']['power_t'])
        max_iterV = int(session['modelArguments']['max_iter'])
        shuffleV = session['modelArguments']['shuffle']
        if(shuffleV == 'True'):
            shuffleV = True
        elif(shuffleV == 'False'):
            shuffleV = False
        tolV = float(session['modelArguments']['tol'])
        model = MLPClassifier(solver=solverV, alpha=alphaV, hidden_layer_sizes=hidden_layer_sizesV, activation=activationV, learning_rate=learning_rateV, learning_rate_init=learning_rate_initV, random_state=1, batch_size=batch_sizeV, power_t=power_tV, shuffle=shuffleV, tol=tolV)
    elif(session['model']=="Random Forest"):
        n_estimatorsV = int(session['modelArguments']['n_estimators'])
        criterionV = session['modelArguments']['criterion']
        max_depthV = session['modelArguments']['max_depth']
        if(max_depthV == 'None'):
            max_depthV = None
        elif(max_depthV != 'None'):
            max_depthV=int(max_depthV)
        min_samples_splitV = int(session['modelArguments']['min_samples_split'])
        min_samples_leafV = int(session['modelArguments']['min_samples_leaf'])
        max_leaf_nodesV = session['modelArguments']['max_leaf_nodes']
        if(max_leaf_nodesV == 'None'):
            max_leaf_nodesV = None
        elif(max_leaf_nodesV != 'None'):
            max_leaf_nodesV=int(max_leaf_nodesV)
        min_impurity_splitV = float(session['modelArguments']['min_impurity_split'])
        model = RandomForestClassifier(n_estimators=n_estimatorsV, criterion=criterionV, max_depth=max_depthV, min_samples_split=min_samples_splitV, min_samples_leaf=min_samples_leafV, max_leaf_nodes=max_leaf_nodesV, min_impurity_split=min_impurity_splitV)
    elif(session['model']=="KNN"):
        n_neighborsV = int(session['modelArguments']['n_neighbors'])
        weightsV = session['modelArguments']['weights']
        algorithmV = session['modelArguments']['algorithm']
        leaf_sizeV = int(session['modelArguments']['leaf_size'])
        pV = int(session['modelArguments']['p'])
        model=KNeighborsClassifier(n_neighbors=n_neighborsV, weights=weightsV, algorithm=algorithmV, leaf_size=leaf_sizeV, p=pV)
    elif(session['model']=="Naive Bayes"):
        model = GaussianNB()
    data, model = trainModel(df, session["model"], targetColumn, cols, model)
    return data, model



def getBestModelParameters(df, target, columns, modelName):
    X = df[columns]  # Features
    y = df[target]  # Target variable
    bestScore = 0
    bestParameters = {}
    model_rs = ''
    if(modelName == 'Decision Tree'):
        md = list(range(1,50))
        md.append(None)
        msl = list(range(1,10))
        msl.append(None)
        mss = np.arange(0.1, 1.0, 0.05)
        mss = np.ndarray.tolist(mss)
        mss = mss+list(range(2,10))
        mln = list(range(2,10))
        mln.append(None)
        tree = DecisionTreeClassifier()
        param_dist = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'min_samples_leaf': msl, 'max_depth': md, 'min_samples_split': mss, 'max_leaf_nodes': mln}
        model_rs = RandomizedSearchCV(tree, param_dist, cv=10)
        model_rs.fit(X,y)
    elif(modelName == 'SVM'):
        c = np.arange(1.0,5.0,0.5)
        c = np.ndarray.tolist(c)
        d = list(range(1,10))
        cf = np.arange(0.0, 5.0, 0.5)
        cf = np.ndarray.tolist(cf)
        mi = d+[-1]
        tol = np.arange(0.0, 5.0, 1e-3)
        tol = np.ndarray.tolist(tol)
        s = svm.SVC()
        param_dist = {'C': c, 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': d, 'gamma': ['scale', 'auto'], 'coef0': cf, 'shrinking': [True, False], 'probability': [True, False], 'decision_function_shape': ['ovo','ovr'], 'tol': tol}
        model_rs = RandomizedSearchCV(s, param_dist, cv=10)
        model_rs.fit(X,y)
    elif(modelName == 'Neural Network'):
        a = np.arange(0.0, 5.0, 0.001)
        a = np.ndarray.tolist(a)
        bs = list(range(50,501,50))
        bs.append('auto')
        pt = np.arange(0.0,5.0,0.5)
        pt = np.ndarray.tolist(pt)
        mi = bs
        lr = np.arange(0.0, 5.0, 0.001)
        lr = np.ndarray.tolist(lr)
        tol = np.arange(0.0, 5.0, 1e-4)
        tol = np.ndarray.tolist(tol)
        nn = MLPClassifier()
        param_dist = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': a, 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'learning_rate_init': lr, 'shuffle': [True, False]}
        model_rs = RandomizedSearchCV(nn, param_dist, cv=10)
        model_rs.fit(X,y)
    elif(modelName == 'Random Forest'):
        ne = list(range(50,501,50))
        md = list(range(1,50))
        md.append(None)
        msl = list(range(1,10))
        msl.append(None)
        mss = np.arange(0.1, 1.0, 0.05)
        mss = np.ndarray.tolist(mss)
        mss = mss+list(range(2,10))
        mln = list(range(2,10))
        mln.append(None)
        rf = RandomForestClassifier()
        param_dist = {'n_estimators': ne, 'criterion': ['gini', 'entropy'], 'min_samples_leaf': msl, 'max_depth': md, 'min_samples_split': mss, 'max_leaf_nodes': mln}
        model_rs = RandomizedSearchCV(rf, param_dist, cv=10)
        model_rs.fit(X,y)
    elif(modelName == 'KNN'):
        n = list(range(1,51))
        ls = list(range(1,201))
        p = n
        knn = KNeighborsClassifier()
        param_dist = {'n_neighbors':n, 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': ls, 'p': p}
        model_rs = RandomizedSearchCV(knn, param_dist, cv=10)
        model_rs.fit(X,y)
    elif(modelName == 'Naive Bayes'):
        vs = np.arange(0.0,1e-9, 1e-7)
        nb = GaussianNB()
        param_dist = {'var_smoothing': vs}
        model_rs = RandomizedSearchCV(nb, param_dist, cv=10)
        model_rs.fit(X,y)
    bestScore = model_rs.best_score_
    print(bestScore)
    bestParameters = model_rs.best_params_
    print(bestParameters)
    bestParametersJSON = json.dumps(bestParameters)
    return bestParametersJSON
