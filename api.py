# -*- coding: utf-8 -*-
"""
@author: anantSinghCross
"""
import flask
import json
import numpy as np
from sklearn.externals import joblib
from flask import Flask, render_template, request
from keras.models import model_from_json

app = Flask(__name__)

@app.route("/")
@app.route("/bostonindex")
def index():
	return flask.render_template('bostonIndex.html')

@app.route("/predict",methods = ['POST'])
def make_predictions():
    if request.method == 'POST':
        a = request.forn.get('crim')
        b = request.forn.get('zn')
        c = request.forn.get('indus')
        d = request.forn.get('chas')
        e = request.forn.get('nox')
        f = request.forn.get('rm')
        g = request.forn.get('age')
        h = request.forn.get('dis')
        i = request.forn.get('rad')
        j = request.forn.get('tax')
        k = request.forn.get('ptratio')
        l = request.forn.get('b')
        m = request.forn.get('lstat')
        
        X = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
        pred = loaded_model.predict(X)
        return flask.render_template('predictPage.html' , response = pred[0][0])
        
        
if __name__ == '__main__':
    json_file = open("model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    app.run(host='0.0.0.0', port=8001, debug=True)
