from plotly import graph_objects as go
import pandas as pd
import json
import plotly
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

def create_hist(current_user, col):
    data = readFile(current_user)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[col]))
    fig.update_layout(xaxis_title=col)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def create_scatter(current_user, col1, col2):
    data = readFile(current_user)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[col1], y=data[col2], mode='markers'))
    fig.update_layout(xaxis_title=col1, yaxis_title=col2)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def create_barplot(current_user, col):
    data = readFile(current_user)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[col]))
    fig.update_layout(xaxis_title=col)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def create_pie(current_user, col):
    data = readFile(current_user)
    fig = go.Figure()
    fig.update_layout(xaxis_title=col)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
