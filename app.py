# Import Necessary Libraries
print('test')
import models
import pickle
import numpy as np
import sqlalchemy
import pandas as pd
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from flask import Flask, jsonify, render_template, redirect, url_for, request
from Data.config import sql_pass
from sklearn.model_selection import train_test_split
import os

#################################################
# Database Setup
#################################################
engine = create_engine(f'postgresql://postgres:{sql_pass}@localhost/breast_cancer1')
conn=engine.connect()
#print(conn.execute("SELECT * from cancerdata"))

# Reflect an existing database into a new model
Base = automap_base()

# Reflect the tables
Base.prepare(engine, reflect=True)


#################################################
# Flask Setup
#################################################
app = Flask(__name__)

# Load the model from disk
loaded_model = pickle.load(open('BCData/randomforest1.sav', 'rb'))
###########################randomforests1.sav######################
# Flask Routes
#################################################

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = loaded_model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 1:
        prediction_text = 'a malignant tumor'
    else:
        prediction_text = 'a benign tumor'
    
    return render_template('index.html', prediction_text=prediction_text)

@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = loaded_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
