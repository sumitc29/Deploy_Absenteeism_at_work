from flask import Flask, render_template, request
import pandas as pd
import scipy
import os
import pickle
import numpy as np


app = Flask(__name__)

def main_fun(data):

    path = os.path.join(app.instance_path, 'features_to_use.sav')
    with open(path , "rb") as input_file:
      required_data = pickle.load(input_file)

    path = os.path.join(app.instance_path, 'ensamble_model.sav')
    with open(path , "rb") as input_file:
      model = pickle.load(input_file)

    path = os.path.join(app.instance_path, 'ohe_object.sav')
    with open(path , "rb") as input_file:
      ohe = pickle.load(input_file)

    path = os.path.join(app.instance_path, 'pca_object.sav')
    with open(path , "rb") as input_file:
      pca = pickle.load(input_file)


    #data  = input(f'enter {required_data} for the prediction saperated bu comma')
    data= data.split(",")
    new_data = []
    for each in data:
      try:
        new_data.append(int(each.strip()))
      except:
        new_data.append(float(each.strip()))


    """getting ohe transform"""
    try:
      out ="Estimated Absenteeism at work is " + str(model.predict(pca.transform(ohe.transform(pd.DataFrame(new_data).T).toarray()))[0]) + " hours"
    except:
      out = "unable to get result please chech data, it may e out of bound"

    return out



@app.route('/')
def my_form():
    return render_template("home.html")

    
@app.route('/', methods=['POST'])
def my_form_post():
    data = request.form['data']
    print(data)
    out = main_fun(data)
    return out
    
    

if __name__ == '__main__':
    app.run()