import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from zipfile import ZipFile

plt.style.use('fivethirtyeight')

# URL de l'API
api_url = "https://mon-api-ef236947dddf.herokuapp.com"

@st.cache_data
def load_data():
    z = ZipFile("default_risk.zip")
    data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding='utf-8')

    z = ZipFile("X_sample.zip")
    sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding='utf-8')
    
    description = pd.read_csv("features_description.csv", 
                              usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')

    target = data.iloc[:, -1:]

    return data, sample, target, description

# Charger les données
data, sample, target, description = load_data()
id_client = sample.index.values

def main():
    #######################################
    # SIDEBAR
    #######################################
    
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sélection de l'ID du client
    st.sidebar.header("**General Info**")
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

    # Affichage de l'ID du client sélectionné
    st.write("Customer ID selection :", chk_id)

    # Affichage des informations du client
    st.header("**Customer information display**")

    if st.checkbox("Show customer information ?"):
        infos_client = data[data.index == int(chk_id)]
        st.write("**Gender : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"] / 365)))
        st.write("**Family status : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Number of children : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

    #######################################
    # Analyse de la solvabilité du client
    #######################################

    st.header("**Customer solvency analysis**")

    # Appel à l'API pour obtenir la prédiction
    try:
        response = requests.get(f"{api_url}/predict_from_id", params={"client": chk_id})
        response.raise_for_status()
        prediction = response.json()

        st.write("**Default probability : **{:.2f} %".format(prediction["prediction_proba"] * 100))
        if prediction["prediction"] == 1:
            st.markdown("<h3 style='color: red;'>Loan is not granted</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>Loan is granted</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error fetching prediction: {e}")

    #######################################
    # Nuage de points avec mise en évidence
    #######################################

    st.header("**Income vs Age Scatterplot**")

    data_sk = data.reset_index(drop=False)
    data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH'] / 365).round(1)

    fig = px.scatter(data_sk, 
                     x='DAYS_BIRTH', 
                     y="AMT_INCOME_TOTAL", 
                     size="AMT_INCOME_TOTAL", 
                     color='CODE_GENDER',
                     hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

    # Mettre en évidence le client sélectionné
    client_data = data_sk[data_sk["SK_ID_CURR"] == int(chk_id)]
    fig.add_scatter(x=client_data["DAYS_BIRTH"], 
                    y=client_data["AMT_INCOME_TOTAL"], 
                    mode="markers", 
                    marker=dict(size=15, color='red'), 
                    name="Selected Client")

    fig.update_layout({'plot_bgcolor': '#f0f0f0'}, 
                      title={'text': "Income vs Age", 'x': 0.5, 'xanchor': 'center'}, 
                      title_font=dict(size=20, family='Arial'))

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
