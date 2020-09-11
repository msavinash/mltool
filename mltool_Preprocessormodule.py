import pandas as pd
import numpy as np

from OneHotEncode.OneHotEncode import *
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

def dataclean(df,colstoremv):
    df.drop(colstoremv, axis = 1, inplace = True)
    return df

def getCatColumns(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat = list(set(cols) - set(num_cols))
    return cat

def ReplaceNaN(df,dict1):
    for col in dict1:
        if(col in df.columns):
            if(dict1[col]=='Median'):
                median = df[col].median()
                df[col].fillna(median, inplace=True)
            elif(dict1[col]=='Mean'):
                mean = df[col].mean()
                df[col].fillna(mean, inplace=True)
            elif(dict1[col]=='Mode'):
                mode= df[col].mode()
                df[col].fillna(mode, inplace=True)
            else:
                df.dropna(subset=[col],inplace=True)
            print(col)
            print(dict1[col])
    return df

def Normalise(df,scaler):
    if(scaler=='MinMaxScaler'):
        df=(df-df.min())/(df.max()-df.min())
        return df
    elif(scaler == 'zscore'):
        df=(df-df.mean())/(df.std())
        return df
    else:
        return df

def handleUncertain(data, log, report):
    temp = report["uncertain"].copy()
    for i in temp:
        data[i] = data[i].str.strip().replace('', np.nan)
        data[i] = data[i].apply(pd.to_numeric, errors='coerce')
        if data[i].isna().sum() == len(data):
            pass
        else:
            report[i]['na'] = data[i].isna().sum() / len(data)
            report[i]['type'] = "num"
            report["uncertain"].remove(i)
            log.append("Modified report to accomodate " + i)
    return data, log, report

def preprop1(data, to_drop=[], na_thresh=0.25):
    log = []
    for i in to_drop:
        data = data.drop(i, axis=1)
        log.append("Dropped column " + i + " as requested")

    report = getStats(data)
    replace_index = {}
    for i in report:
        if i == "uncertain":
            data, log, report = handleUncertain(data, log, report)
        elif i in report["uncertain"]:
            data = data.drop(i, axis=1)
            log.append("Dropped column " + i + " due to uncertainity")
        elif report[i]["type"] == 'num' and report[i]["na"] > 0 and report[i]["na"] < na_thresh :
            data[i] = data[i].fillna(data[i].mean())
            log.append(i + " : replaced Nan/Na with mean")

        elif (report[i]["type"] == 'num' or report[i]["type"] == 'cat') and report[i]["na"] >= na_thresh :
            data = data.drop(i, axis=1)
            log.append("Dropped column " + i + " as Nan/Na ratio (" + str(report[i]["na"]) + ") > " + str(na_thresh))
        elif report[i]["type"] == 'cat' and report[i]["na"] > 0 and report[i]["na"] < na_thresh :
            log.append(i + " has Na/empty percentage of " + str(report[i]["na"]) + " < " + str(na_thresh) + " but is categorical.")
        elif report[i]["type"] == 'cat' and i in data.columns:
            replace_index[i] = getEncodings(data[i])
            log.append("Encoded column " + i)

    data = data.replace(replace_index)
    return data, log, replace_index

def getEncodings(o):
    o = list(set(o))
    o = sorted(o)
    enc = {cls: ind for ind, cls in enumerate(o)}
    return enc

def getStats(data):
    cols = data.columns
    report = {}
    report['uncertain'] = []
    for i in cols:
        report[i] = {}
        report[i]["na"] = data[i].isna().sum() / len(data)
        if "unique" in str(data[i].describe()):
            if len(data[i].unique()) <  0.25 * len(data):
                report[i]["type"] = "cat"
                report[i]["uniq"] = data[i].unique()
                report[i]["uniq_no"] = len(data[i].unique())
            else:
                report["uncertain"].append(i)
        else:
            report[i]["type"] = "num"
    return report

def preprop(data):
    d1, log, encs = preprop1(data)
    d1 = d1.dropna()
    d2,log2,enc2=preprop1(d1)
    normalized_df=(d2-d2.mean())/(d2.std())
    d3,log3,enc3=preprop1(normalized_df)
    return d3,log,encs
