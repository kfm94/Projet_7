#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import flask 
from flask import Flask, render_template, request, jsonify

import pickle


# In[3]:


# création d'une extension flask 
app = Flask(__name__)


# In[ ]:





# In[4]:


# Charger le modèle plus tard si nécessaire
with open('lightgbm_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


# In[7]:

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = loaded_model.predict_proba(pd.DataFrame(data))[:, 1].item()
    return jsonify({"prediction": prediction})



#s'il y a un score demandé remplacé predict_from_id par predict_proba
@app.route('/predict_from_id', methods=['GET'])
def predict_from_id():
    train = pd.read_csv('X_train.csv')
    id_client = request.args.get("client")
    train = train.iloc[:, 0:572]
    data_client = train.iloc[[id_client]]
    
    prediction = loaded_model.predict(data_client)
    prediction = prediction[0]
    return jsonify({"prediction": prediction})



@app.route('/', methods=['GET'])
def home():
    return "Bonjour"
 
    

if __name__ == '__main__':
    app.run(debug = True, port = 51000)


