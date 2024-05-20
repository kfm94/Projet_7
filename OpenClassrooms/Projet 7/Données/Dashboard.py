import streamlit as st
import requests
import matplotlib.pyplot as plt


url = "http://127.0.0.1:51000"

# Function to get customer details
def get_customer_details(cust_id):
    response = requests.get(f"{url}/details/id={cust_id}")
    data = response.json()
    return data

# Function to get customer prediction
def get_customer_prediction(cust_id):
    response = requests.get(f"{url}/predict_from_id?client={cust_id}")
    data = response.json()
    return data

# Function to get feature importance for selected customer
def get_feature_importance(cust_id):
    response = requests.get(f"{url}/details/id={cust_id}")
    data = response.json()
    if 'client_shap_values' in data:
        return data['client_shap_values']
    else:
        return None

# Function to get distribution data for selected feature
def get_feature_distribution(feature_name):
    response = requests.get(f"{url}/distribution/feature={feature_name}")
    distribution_data = response.json()
    if 'accepted' in distribution_data and 'rejected' in distribution_data:
        return distribution_data
    else:
        return None

# Main function to display dashboard
def main():
    # Get customer ID from user input
    cust_id = st.number_input('Enter Customer ID:', min_value=0, max_value=999, value=0)

    # Display customer details
 #   st.write("### Customer Details:")
#    details = get_customer_details(cust_id)
#    st.write(details)

    # Display customer prediction
    st.write("### Prediction:")
    prediction = get_customer_prediction(cust_id)
    st.write(prediction)

    # Display feature importance for selected customer
#    st.write("### Feature Importance for Selected Customer:")
#    feature_importance = get_feature_importance(cust_id)
#    if feature_importance:
#        st.write(feature_importance)
#    else:
#        st.write("Feature importance data not available for this customer.")

    # Dropdown to select feature for distribution analysis
    feature_name = st.selectbox("Select feature for distribution analysis:", ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "NAME_FAMILY_STATUS_Married", "NAME_INCOME_TYPE_Working", "AMT_INCOME_TOTAL", "PAYMENT_RATE", "DAYS_BIRTH", "DAYS_EMPLOYED"])

    # Display distribution of selected feature
 #   st.write("### Distribution of Features:")
#    distribution_data = get_feature_distribution(feature_name)
#    if distribution_data:
#        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
#        axes[0].hist(distribution_data['accepted'], bins=30, color='blue', alpha=0.5, label='Accepted')
#        axes[0].set_title('Distribution on Accepted Credits')
#        axes[0].set_xlabel(feature_name)
#        axes[0].set_ylabel('Frequency')
#        axes[0].legend()

#        axes[1].hist(distribution_data['rejected'], bins=30, color='red', alpha=0.5, label='Rejected')
#        axes[1].set_title('Distribution on Rejected Credits')
#        axes[1].set_xlabel(feature_name)
#        axes[1].set_ylabel('Frequency')
#        axes[1].legend()

#        st.pyplot(fig)
#    else:
#        st.write("Distribution data not available for this feature.")
#
if __name__ == "__main__":
    main()
