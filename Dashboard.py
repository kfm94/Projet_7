import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from zipfile import ZipFile
import pickle

plt.style.use('fivethirtyeight')

# URL de l'API
#url = "http://127.0.0.1:51000"
#url = "https://proj7-4e678206e3eb.herokuapp.com/"
url = "https://mon-api-ef236947dddf.herokuapp.com/"

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

@st.cache_data
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets

def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client

@st.cache_data
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"] / 365), 2)
    return data_age

@st.cache_data
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income

@st.cache_data
def load_prediction(id, _url):  # Requête API pour la prédiction
    try:
        response = requests.get(f"{_url}/predict_from_id?client={id}")
        response.raise_for_status()  # Vérifie les erreurs HTTP
        data = response.json()
        return data["prediction"], data["prediction_proba"]
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erreur HTTP: {http_err}")
        return None, None
    except requests.exceptions.RequestException as req_err:
        st.error(f"Erreur de connexion: {req_err}")
        return None, None
    except ValueError as json_err:
        st.error(f"Erreur de décodage JSON: {json_err}")
        st.error(f"Réponse de l'API : {response.text}")
        return None, None

# Charger les données et les modèles
data, sample, target, description = load_data()
id_client = sample.index.values

def main():
    #######################################
    # SIDEBAR
    #######################################
    
    # Titre du dashboard
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

    # Informations générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)

    # Affichage des informations dans la barre latérale
    st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)
    st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)
    st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)

    # Diagramme en secteurs
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

    # Affichage de l'ID du client sélectionné
    st.write("Customer ID selection :", chk_id)

    # Affichage des informations du client
    st.header("**Customer information display**")

    if st.checkbox("Show customer information ?"):
        infos_client = identite_client(data, chk_id)
        st.write("**Gender : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"] / 365)))
        st.write("**Family status : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Number of children : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

        # Distribution de l'âge
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor='k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)

        st.subheader("*Income (USD)*")
        st.write("**Income total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Credit amount : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))

        # Afficher un nuage de points coloré avec le client sélectionné
        st.subheader("**Customer Scoring Visualization**")

        # Extraire les caractéristiques pour le nuage de points
        x = data["AMT_INCOME_TOTAL"]
        y = data["AMT_CREDIT"]

        # Création du nuage de points
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, c='blue', label='Other clients', alpha=0.5)

        # Tracer le point du client sélectionné en rouge
        client_data = infos_client.iloc[0]
        ax.scatter(client_data["AMT_INCOME_TOTAL"], client_data["AMT_CREDIT"], c='red', label='Selected Client', edgecolors='black')

        # Ajouter des labels
        ax.set_xlabel('Income Total (USD)')
        ax.set_ylabel('Credit Amount (USD)')
        ax.set_title('Income vs Credit Amount (Scoring)')

        # Légende
        ax.legend(loc='best')
        st.pyplot(fig)

        # Prédiction du risque de défaut
        st.subheader("**Risk prediction**")

        # Appel à l'API pour obtenir la prédiction
        prediction, prediction_proba = load_prediction(chk_id, url)
        
        if prediction is not None and prediction_proba is not None:
            st.write(f"Risk prediction: {'Default' if prediction == 1 else 'No Default'}")
            st.write(f"Probability of default: {prediction_proba:.2f}")
        else:
            st.warning("Erreur lors de la récupération de la prédiction.")

if __name__ == "__main__":
    main()
