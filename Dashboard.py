import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use('fivethirtyeight')

# URL de l'API
url = "http://127.0.0.1:51000"

# Charger les données statiques
@st.cache_data
def load_data():
    data = pd.read_csv("default_risk.csv", index_col='SK_ID_CURR', encoding='utf-8')
    sample = pd.read_csv("X_sample.csv", index_col='SK_ID_CURR', encoding='utf-8')
    description = pd.read_csv("features_description.csv", 
                              usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
    return data, sample, description

# Appel à l'API pour obtenir une prédiction
def get_prediction(client_id):
    response = requests.get(f"{url}/predict_from_id", params={"client": client_id})
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Impossible de récupérer la prédiction"}

# Appel à l'API pour obtenir les détails d'un client
def get_client_details(client_id):
    response = requests.get(f"{url}/details/id={client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Impossible de récupérer les détails du client"}

# Charger les données
data, sample, description = load_data()
id_client = sample.index.values

# Streamlit
def main():
    st.sidebar.header("**General Info**")
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    # Appels à l'API
    prediction_data = get_prediction(chk_id)
    client_data = get_client_details(chk_id)

    # Informations générales sur le client
    st.header("**Customer Information Display**")
    if client_data.get("error"):
        st.error(client_data["error"])
    else:
        st.write(f"**Client ID:** {chk_id}")
        st.write(f"**Gender:** {client_data['CODE_GENDER']}")
        st.write(f"**Age:** {int(-client_data['DAYS_BIRTH'] / 365)} years")
        st.write(f"**Income Total:** {client_data['AMT_INCOME_TOTAL']}")

    # Analyse de la solvabilité
    st.header("**Customer Solvency Analysis**")
    if prediction_data.get("error"):
        st.error(prediction_data["error"])
    else:
        st.write(f"**Default Probability:** {prediction_data['prediction_proba']:.2f}")
        if prediction_data["prediction"] == 1:
            st.markdown("<h3 style='color: red;'>Loan is not granted</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>Loan is granted</h3>", unsafe_allow_html=True)

    # Relation Âge/Revenu avec mise en évidence du client
    st.header("**Relationship Age / Income Total**")
    data_sk = data.reset_index()
    data_sk["AGE"] = -data_sk["DAYS_BIRTH"] / 365
    fig = px.scatter(data_sk, x="AGE", y="AMT_INCOME_TOTAL",
                     size="AMT_INCOME_TOTAL", color="CODE_GENDER",
                     hover_data=["NAME_FAMILY_STATUS", "CNT_CHILDREN", "NAME_CONTRACT_TYPE"],
                     title="Relationship Age / Income Total")
    
    # Ajouter un point pour le client sélectionné
    client_age = -client_data["DAYS_BIRTH"] / 365
    client_income = client_data["AMT_INCOME_TOTAL"]
    fig.add_scatter(x=[client_age], y=[client_income], mode='markers', 
                    marker=dict(size=12, color='red'), name='Selected Client')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
