import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application


#import ridge model and standardscaler

ridge_model = pickle.load(open('models/reg (1).pkl','rb'))
stand_scaler = pickle.load(open('models/stand.pkl','rb'))

#creating homepage
@app.route("/")
def index():
      return render_template('index.html')

#predict my model data
@app.route("/predictdata",methods=['POST','GET'])
def predict_datapoint():
     if request.method == 'POST':
          #read the data
          Temperature = float(request.form.get('Temperature'))
          RH = float(request.form.get('RH'))
          Ws = float(request.form.get('Ws'))
          Rain = float(request.form.get('Rain'))
          FFMC = float(request.form.get('FFMC'))
          DMC = float(request.form.get('DMC'))
          ISI = float(request.form.get('ISI'))
          Classes = float(request.form.get('Classes'))
          Region = float(request.form.get('Region'))
          
          #For every new data it will transform 
          new_data_scaled = stand_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
          result = ridge_model.predict(new_data_scaled)

          return render_template('home.html',results=result[0])
     else:
          return render_template('home.html')
   

if __name__=="__main__":
   app.run(host="0.0.0.0") 