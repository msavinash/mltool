from flask import Flask, jsonify
from flask_cors import CORS
from flask import request, session
from flask import render_template
from time import sleep
from random import random

import json
import os
import pickle
from fpdf import FPDF
import zipfile
from shutil import copyfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from OneHotEncode.OneHotEncode import *
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import mltool_MLmodule as myML
import mltool_Graphmodule as mygraph
import auth
import mltool_Preprocessormodule as preprocessor


# New
from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)
from flask import send_from_directory, send_file
#New end


# configuration
DEBUG = True

USERS_DIR = "USERS/"
DATASET_UPLOAD_FOLDER = '/datasets'


# instantiate the app
app = Flask(__name__)
app.secret_key = 'super secret string'  # Change this!
app.config.from_object(__name__)

#New
jwt = JWTManager(app)

# enable CORS
CORS(app, resources={r'/*': {'origins': 'http://localhost:8080'}})
CORS(app, resources={r'/*': {'origins': '*'}})


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



def updateCurrentSession(current_user, session):
    pickle.dump(session, open(USERS_DIR+current_user+"/session.pkl","wb"))

def convert_pickle_to_json(file_path):
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
        data = pickle.load(fpkl)
        data=data.tolist()
        json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4)
        return json

def createPDF(current_user):
    session = getCurrentSession(current_user)
    model = session['model']
    print("\n"*5,"crpdf: ", model)
    details = session['data']
    print("\n"*5,details,"\n"*5)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial",'U', size=18)
    pdf.cell(200, 10, txt="ML Trained Model Report", ln=3, align="C")
    pdf.cell(ln=1, h=5.0, align='L', w=0, txt=" ", border=0) #line break
    pdf.set_font("Arial", size=15)
    pdf.cell(200,10,txt= "Model Chosen: %s" % model, ln=4, align="C")

    pdf.cell(ln=1, h=5.0, align='L', w=0, txt=" ", border=0)
    pdf.cell(ln=1, h=5.0, align='L', w=0, txt=" ", border=0)
    pdf.cell(200,10,txt= "Accuracy: %s" % details["Accuracy"], ln=7, align="C")
    pdf.cell(200,10,txt= "f1-score: %s" % details["f1-score"], ln=8, align="C")
    pdf.cell(200,10,txt= "Precision: %s" % details["Precision"], ln=9, align="C")
    pdf.cell(200,10,txt= "Recall: %s" % details["Recall"], ln=10, align="C")
    pdf.cell(200,10,txt= "Confusion Matrix: %s"% details["Confusion Matrix"], ln=11, align="C")
    pdf.output(USERS_DIR+current_user+"/TrainedModel/Trained Model Report.pdf")

def createReadme(current_user):
    print("\n"*5,"Inside createReadme()","\n"*5)
    with open(USERS_DIR+current_user+"/TrainedModel/Readme.md","w+") as file:
        print("\n"*5,"Inside file!","\n"*5)
        file.write("This folder contains a pickle file called trainedModel.pkl\n")
        file.write("The trained model is saved in this pkl file.\n\n")
        file.write("The instructions to use the pkl file are as follows:\n\n")
        file.write("model = pickle.load(open(trainedModel.pkl, 'rb'))\nresult = model.score(X_test, Y_test)\nprint(result)\n")

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),os.path.relpath(os.path.join(root,file), os.path.join(path,'..')))


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/test', methods=['GET'])
def p():
    return jsonify('Ping pong')

@app.route("/signup/check-username", methods=["POST"])
def checkUsername():
    requestJSON = request.json
    resp = auth.checkUsername(requestJSON)
    return resp


@app.route("/login", methods=["POST"])
def login():
    requestJSON = request.json
    resp = auth.checkUser(requestJSON)
    print(resp)
    if resp:
        access_token = create_access_token(identity=requestJSON["username"])
        return jsonify({"login":"success", "auth":access_token})
    else:
        return jsonify({"login":"fail"})


@app.route("/signup", methods=["POST"])
def signup():
    requestJSON = request.json
    resp = auth.addUser(requestJSON)
    if resp:
        session["username"] = requestJSON["username"]
        access_token = create_access_token(identity=requestJSON["username"])
        return jsonify({"login":"success", "auth":access_token})
    else:
        return jsonify({"login":"fail"})


@app.route("/sessiontest", methods=["POST"])
def sessiontest():
    # requestJSON = request.json
    # resp = auth.details(session)
    print(flask_login.current_user.id)
    return '0'


@app.route("/getUserDetails", methods=["POST"])
@jwt_required
def getDetails():
    print("\n"*5,"in getDetails","\n"*5)
    requestJSON = request.json
    current_user = get_jwt_identity().lower()
    if requestJSON["type"] == "primary":
        details = auth.details(current_user)
        print("\n"*5,"details:",details,"\n"*5)
        resp = {
                "username":details[0],
                "emailID":details[2]
                }
        print(resp, "\n"*10)
        return jsonify(resp)
    elif requestJSON["type"] == "all":
        details = auth.details(current_user)
        print(details)
        resp = {
                "username":details[0],
                "emailID":details[2],
                "firstName":details[3],
                "lastName":details[4],
                "address":details[5],
                "city":details[6],
                "country":details[7],
                "postalCode":details[8],
                "aboutMe":details[9]
                }
        return jsonify(resp)

@app.route('/updateProfile',methods=['POST'])
@jwt_required
def updateProfile():
    current_user = get_jwt_identity().lower()
    requestJSON = request.json
    print("USER DETAILS:", requestJSON)
    d = {}
    for i in requestJSON.keys():
        if requestJSON[i]:
            d[i] = requestJSON[i]
    success = auth.updateDetails(current_user, d)
    return "DONE"

@app.route('/getAllDatasets', methods=['GET'])
@jwt_required
def getAllDatasets():
    current_user = get_jwt_identity().lower()
    d = os.listdir(USERS_DIR+current_user+DATASET_UPLOAD_FOLDER)
    print("\n"*5,d,"\n"*5)
    datasets = {}
    for i,dataset in enumerate(d):
        datasets[str(i)] = dataset
    print(datasets)
    return datasets


@app.route('/uploadDataset',methods=['GET','POST'])
@jwt_required
def uploadDataset():
    if(request.method=='POST'):
        file=request.files['file']
        print("\n"*5,"In uploadDataset","\n"*5)
        print(file)
        print(file.filename)
        current_user = get_jwt_identity().lower()
        session = getCurrentSession(current_user)
        session['filename'] = file.filename
        file.save(os.path.join(USERS_DIR+current_user+DATASET_UPLOAD_FOLDER, session['filename']))
        updateCurrentSession(current_user, session)
        print("\n"*5,"\n"*5)
        if file:
            df = readFile(current_user)
            print(df)
            print(df.head())
            print("\n"*5,"\n"*5)
            return jsonify("Dataset uploaded successfully.")
        else:
            return jsonify("Please attach file for analysis<br><br><a href='/'>Back</a>")
    else:
        return jsonify("Dataset not attached")


@app.route('/selectDataset',methods=['GET','POST'])
@jwt_required
def selectDataset():
    if(request.method=='POST'):
        current_user = get_jwt_identity().lower()
        session = getCurrentSession(current_user)
        print("\n"*5,request,"\n"*5)
        post_data=request.json
        session['filename'] = post_data['activeDataset']
        updateCurrentSession(current_user, session)
        print("\n"*5,"\n"*5)
        if 'filename' in session:
            df = readFile(current_user)
            print(df)
            print(df.head())
            print("\n"*5,"\n"*5)
            return jsonify("Dataset selected successfully.")
        else:
            return jsonify("Please attach file for analysis<br><br><a href='/'>Back</a>")
    else:
        return jsonify("Dataset not selected")


@app.route('/getColumns',methods=['GET'])
@jwt_required
def getColumns():
    print("\n"*5,"In getColumns()","\n"*5)
    current_user = get_jwt_identity().lower()
    df = readFile(current_user)
    c = {}
    for index, column in enumerate(df.columns):
        c[str(index)] = column
    print(c,"\n"*5)
    return c


@app.route('/getNaNcolumns',methods=['GET'])
@jwt_required
def getNaNcolumns():
    current_user = get_jwt_identity().lower()
    df=readFile(current_user)
    c = {}
    for column in df.columns[df.isna().any()].tolist():
        c[column]='k'

    return c


@app.route('/recPreProp',methods=['POST'])
@jwt_required
def recPreProp():
    if(request.method=='POST'):
        post_data=request.get_json();
        print("Target column = ",post_data['target'])
        current_user = get_jwt_identity().lower()
        session = getCurrentSession(current_user)
        filename=session['filename']
        df=readFile(current_user)
        le = LabelEncoder()
        catCol = preprocessor.getCatColumns(df)
        if(post_data['target'] in catCol):
            df[post_data['target']]=le.fit_transform(df[post_data['target']].astype(str))
        dftarget = df[post_data['target']]
        df.drop(post_data['target'], axis=1, inplace=True)
        print(dftarget)
        df,log,encs = preprocessor.preprop(df)
        df[post_data['target']] = dftarget
        df.to_csv(os.path.join(USERS_DIR+current_user+DATASET_UPLOAD_FOLDER, session['filename']),index=False)
        return jsonify("Cleaned successfully.")

@app.route('/removeFeatures',methods=['POST'])
@jwt_required
def removeFeatures():
    if(request.method=='POST'):
        post_data=request.get_json();
        print(post_data)
        current_user = get_jwt_identity().lower()
        session = getCurrentSession(current_user)
        filename=session['filename']
        df=readFile(current_user)
        dftarget = df[post_data['targetC']]
        df.drop(post_data['targetC'], axis=1, inplace=True)
        colstoremv=post_data['remove_feature']
        print(colstoremv)
        df = preprocessor.dataclean(df,colstoremv)
        categorical = preprocessor.getCatColumns(df)
        print(categorical)
        if(post_data['encoder']=='One-Hot'):
            df,dropped_cols,all_new_cols,new_col_dict = OneHotEncode(df,categorical,check_numerical=False,max_var=20)
            print(all_new_cols)
            print(new_col_dict)
        else:
            le = LabelEncoder()
            print("Here before Label encoding")
            for i in categorical:
                try:
                    df[i]=le.fit_transform(df[i].astype(str))
                    print("Label encoding done for",i)
                except:
                    print("couldnt label encode for ",i)
        dict1 =post_data['NaN']
        df = preprocessor.ReplaceNaN(df,post_data['NaN'])
        scaler=post_data['scaler']
        df = preprocessor.Normalise(df,scaler)
        df[post_data['targetC']] = dftarget
        df.to_csv(os.path.join(USERS_DIR+current_user+DATASET_UPLOAD_FOLDER, session['filename']),index = False)
        print(df)
        return jsonify("Cleaned successfully.")


@app.route('/getPreProcessedDataset', methods=['GET'])
@jwt_required
def getPreProcessedDataset():
    current_user = get_jwt_identity().lower()
    session = getCurrentSession(current_user)
    filename = session['filename']
    if not os.path.exists(USERS_DIR+current_user+'/static'):
        os.makedirs(USERS_DIR+current_user+'/static')
    copyfile(USERS_DIR+current_user+'/datasets/'+filename,USERS_DIR+current_user+'/static/'+filename)
    return send_from_directory(USERS_DIR+current_user+'/static', filename=filename, mimetype='text/csv', as_attachment=True, cache_timeout=30*60)


@app.route('/getBestParameters',methods=['GET','POST'])
@jwt_required
def getBestParameters():
    if(request.method=='POST'):
        current_user = get_jwt_identity().lower()
        post_data=request.get_json();
        session = getCurrentSession(current_user)
        session['model']=post_data['selectedModel']
        session['targetColumn']=post_data['targetColumn']
        session['columns']=post_data['columns']
        updateCurrentSession(current_user, session)
    df = readFile(current_user)
    bp = {}
    bp = myML.getBestModelParameters(df, session['targetColumn'], session['columns'], session['model'])
    return bp


@app.route('/setModel',methods=['GET','POST'])
@jwt_required
def setModel():
    print("\n"*5,"In setModel()","\n"*5)
    if(request.method=='POST'):
        print("In POST method of setModel")
        post_data=request.get_json()
        current_user = get_jwt_identity().lower()
        session = getCurrentSession(current_user)
        session['model']=post_data['selectedModel']
        session['targetColumn']=post_data['targetColumn']
        session['columns']=post_data['columns']
        session['modelArguments']=post_data['modelArguments']
        print("\n"*5,"Session: ",session,"\n"*5)
        df = readFile(current_user)
        print("Here\n\n", session["model"])
        data, model = myML.setupModel(session, current_user)
        if not os.path.exists(USERS_DIR+current_user+'/TrainedModel'):
            os.makedirs(USERS_DIR+current_user+'/TrainedModel')
        pickle.dump(model, open(USERS_DIR+current_user+"/TrainedModel/trainedModel.pkl", "wb"))
        session['data']=data
        updateCurrentSession(current_user, session)
        return "Model set successful"
    return "Error in set model"


@app.route('/getReport',methods=['GET'])
@jwt_required
def getReport():
    print("\n"*5,"In getReport()","\n"*5)
    current_user = get_jwt_identity().lower()
    if not os.path.exists(USERS_DIR+current_user+'/static'):
        os.makedirs(USERS_DIR+current_user+'/static')
    if os.path.isfile(USERS_DIR+current_user+"/TrainedModel/trainedModel.pkl"):
        createPDF(current_user)
        print("\n"*5,"Creating readme","\n"*5)
        createReadme(current_user)
        print("Creating ZIP","\n"*10)
        zipf = zipfile.ZipFile(USERS_DIR+current_user+'/static/TrainedModel.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(USERS_DIR+current_user+'/TrainedModel/', zipf)
        zipf.close()
    else:
        raise FileNotFoundError
    session = getCurrentSession(current_user)
    data = session['data']
    return data


@app.route("/plot", methods=["POST"])
@jwt_required
def plot():
    requestJSON = request.get_json()
    graph = "Not possible"
    current_user = get_jwt_identity().lower()
    session = getCurrentSession(current_user)
    if requestJSON["graphType"] == "histogram":
        column = requestJSON["columns"][0]
        graph = mygraph.create_hist(current_user, column)
    elif requestJSON["graphType"] == "scatterplot":
        column1 = requestJSON["columns"][0]
        column2 = requestJSON["columns"][1]
        graph = mygraph.create_scatter(current_user, column1, column2)
    elif requestJSON["graphType"] == "barplot":
        column = requestJSON["columns"][0]
        graph = mygraph.create_barplot(current_user, column)
    elif requestJSON["graphType"] == "piechart":
        column = requestJSON["columns"][0]
        graph = mygraph.create_pie(current_user, column)
    return graph


@app.route('/getTrainedModelZip')
@jwt_required
def getTrainedModelZip():
    current_user = get_jwt_identity().lower()
    print("Sending Zip...")
    return send_from_directory(USERS_DIR+current_user+'/static', filename='TrainedModel.zip', mimetype='application/zip', as_attachment=True, cache_timeout=30*60)

if __name__ == '__main__':
    # app.run('0.0.0.0',port=8080)
    app.run()
