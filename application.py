from flask import Flask,request,app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application=Flask(__name__)
app=application

scaler=pickle.load(open("Model/standardscaler.pkl",'rb'))
model=pickle.load(open("Model/modelforprediction.pkl",'rb'))

##route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## route for single data point prediction

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""
    
    if request.method=='POST':
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose=int(request.form.get("Glucose"))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreenFunction=float(request.form.get('DiabetesPedigreenFunction'))
        Age=float(request.form.get('Age'))
        
        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreenFunction,Age]])
        predict=model.predict(new_data)
        
        if predict[0]==1:
            result="Diabetic"
        else:
            result="Non-Diabetic"
            
        return render_template("single_prediction.html",result=result)
    else:
        return render_template("home.html")
    

if __name__=="__main__":
    app.run(host='0.0.0.0')