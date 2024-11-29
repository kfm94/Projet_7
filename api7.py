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

@app.route('/predict_from_id', methods=['GET'])
def predict_from_id():
    try:
        id_client = int(request.args.get("client"))
        if id_client not in train.index:
            return jsonify({"error": "L'ID client spécifié est invalide."}), 404
        
        # Extraire les données du client
        data_client = train.iloc[[id_client]]
        data_client = data_client[model_features]
        
        # Prédiction (probabilité)
        prediction_proba = loaded_model.predict_proba(data_client)[:, 1].item()
        
        # Appliquer un seuil pour la classe binaire (0 ou 1)
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        return jsonify({
            "prediction": prediction,
            "prediction_proba": prediction_proba
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/details/id=<int:id_client>', methods=['GET'])
def get_customer_details(id_client):
    try:
        data_client = train.iloc[id_client].to_dict()
        return jsonify(data_client)
    except IndexError:
        return jsonify({"error": "Client ID not found"}), 404

@app.route('/', methods=['GET'])
def home():
    return "Bonjour"

if __name__ == '__main__':
    app.run(debug=True, port=51000)
