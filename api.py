import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Charger le modèle
with open('lightgbm_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Charger les données du client
train = pd.read_csv('X_train.csv')

# Obtenir les colonnes de caractéristiques utilisées pour entraîner le modèle
model_features = train.columns[:572]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame(data)
    
    # Sélectionner les caractéristiques correctes
    input_data = input_data[model_features]
    
    prediction = loaded_model.predict_proba(input_data)[:, 1].item()
    return jsonify({"prediction": prediction})

@app.route('/predict_from_id', methods=['GET'])
def predict_from_id():
    id_client = int(request.args.get("client"))
    data_client = train.iloc[[id_client]]
    
    # Sélectionner les caractéristiques correctes
    data_client = data_client[model_features]
    
    prediction = loaded_model.predict(data_client)
    prediction_proba = loaded_model.predict_proba(data_client)[:, 1].item()
    return jsonify({"prediction": prediction[0], "prediction_proba": prediction_proba})

@app.route('/details/id=<int:id_client>', methods=['GET'])
def get_customer_details(id_client):
    data_client = train.iloc[id_client].to_dict()
    return jsonify(data_client)

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    # Placeholder for feature importance data, replace with actual SHAP values if available
    feature_importance = {
        "client_shap_values": [0.1, 0.2, -0.1]  # Example values, replace with actual SHAP values
    }
    return jsonify(feature_importance)

@app.route('/distribution/feature=<feature_name>', methods=['GET'])
def get_feature_distribution(feature_name):
    # Placeholder for feature distribution data, replace with actual distribution data if available
    distribution_data = {
        "accepted": [0.1, 0.2, 0.3],  # Example values, replace with actual distribution data
        "rejected": [0.3, 0.2, 0.1]   # Example values, replace with actual distribution data
    }
    return jsonify(distribution_data)

@app.route('/', methods=['GET'])
def home():
    return "Bonjour"

if __name__ == '__main__':
    app.run(debug=True, port=51000)
