import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
import pickle

plt.style.use('fivethirtyeight')

# URL de l'API
url = "http://127.0.0.1:51000"

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

@st.cache_resource
def load_model():
    with open('LGBMClassifier.pkl', 'rb') as file: 
        clf = pickle.load(file)
    return clf

@st.cache_data
def load_knn(sample):
    knn = knn_training(sample)
    return knn

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
def load_prediction(sample, id, _clf):  # Ajouter un trait de soulignement devant clf
    X = sample.iloc[:, :-1]
    score = _clf.predict_proba(X[X.index == int(id)])[:, 1]
    return score

@st.cache_data
def load_kmeans(sample, id, _knn):  # Ajouter un trait de soulignement devant knn
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(_knn.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:, 1:].sample(10)

@st.cache_data
def knn_training(sample):
    knn = KMeans(n_clusters=2).fit(sample)
    return knn

# Charger les données et les modèles
data, sample, target, description = load_data()
id_client = sample.index.values
clf = load_model()

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
        st.write("**Credit annuities : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Amount of property for credit : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))

        # Distribution des revenus
        data_income = load_income_population(sample)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor='k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)

        # Relation entre l'âge et les revenus
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH'] / 365).round(1)
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor': '#f0f0f0'}, 
                          title={'text': "Relationship Age / Income Total", 'x': 0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Arial'))
        st.plotly_chart(fig)

    #######################################
    # Analyse de la solvabilité du client
    #######################################

    st.header("**Customer solvency analysis**")

    prediction = load_prediction(sample, chk_id, clf)
    st.write("**Default probability : **{:.2f} %".format(prediction[0]*100))

    if prediction >= 0.5:
        st.markdown("<h3 style='color: red;'>Loan is not granted</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>Loan is granted</h3>", unsafe_allow_html=True)

    #######################################
    # Importance des caractéristiques / description
    #######################################

    st.header("**Features importance / description**")

    if st.checkbox("Show feature importance ?"):
        shap.initjs()
        X = sample.iloc[:, :-1]
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 10))
        shap.summary_plot(shap_values, X)  # Correction ici pour utiliser l'ensemble des valeurs shap
        st.pyplot(fig)

        if st.checkbox("Show feature description ?"):
            list_features = X.columns.to_list()
            feature = st.selectbox('Feature checklist', list_features)
            st.table(description.loc[description.index == feature][:1])

    #######################################
    # Affichage des fichiers clients similaires
    #######################################

    st.header("**Similar customers file display**")

    knn = load_knn(sample)
    knn_df = load_kmeans(sample, chk_id, knn)

    if st.checkbox("Show similar customer files ?"):
        st.dataframe(knn_df)

if __name__ == "__main__":
    main()
