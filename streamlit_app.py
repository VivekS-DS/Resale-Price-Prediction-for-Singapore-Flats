import pickle
import pandas as pd
import streamlit as st
import gdown
import os

import warnings
warnings.filterwarnings('ignore')  # Ignore warnings during execution for cleaner output

# Title and introduction
st.title('Resale Price Prediction for Singapore Flats')
st.write("Welcome to the Singapore Resale Price Predictor! This application is designed to help you estimate "
    "the resale price of a flat based on historical transaction data. Simply input the details of the flat, "
    "and let the machine learning model provide you with an estimated resale price.")

# Function to download the pre-trained model
def load_date():
    # URL for the pre-trained model
    model_url = "https://onedrive.live.com/download?resid=F275F26477A8CF4E%21378506&authkey=!AIpRfUl-iGUr7ik"
    # Output file for saving the downloaded model
    output_file = "realestate.pkl"
    # Downloading the model only if it doesn't exist locally
    if not os.path.exists(output_file):
        gdown.download(model_url, output_file, quiet=False)
    return

# Sidebar for user input
st.sidebar.header('Enter Details')
# User Input
town = st.sidebar.selectbox('Town',('ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                'TOA PAYOH', 'WOODLANDS', 'YISHUN'))
flat_type = st.sidebar.selectbox('flat Type',('2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM',
                                         'MULTI-GENERATION'))
Storey = st.sidebar.selectbox('storey',('Mid_Rise_Building', 'Low_Rise_Building', 'High_Rise_Building'))
floor_area = st.sidebar.slider('floor area in sq.m',30,250,50)
flat_model = st.sidebar.selectbox('flat model',('Improved', 'New Generation', 'DBSS', 'Standard', 'Apartment',
                                           'Simplified', 'Model A', 'Premium Apartment', 'Adjoined flat',
                                           'Model A-Maisonette', 'Maisonette', 'Type S1', 'Type S2',
                                           'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Maisonette',
                                           'Multi Generation', 'Premium Apartment Loft', '2-room', '3Gen'))
flat_age_months = st.sidebar.slider('flat_age_months', 12, 684, 12)
lease_exp = st.sidebar.slider('lease_exp in months', 12, 1200, 50)

# Button to submit user input
user_submit = st.sidebar.button('Submit')
if user_submit:
    load_date() 
    with open("realestate.pkl", 'rb') as f:
        model_rf = pickle.load(f)

    # User input data
    user_data = { 'town' : town,
                  'flat_type' : flat_type,
                  'storey_range': Storey,
                  'floor_area_sqm': floor_area,
                  'flat_model': flat_model,
                  'flat_age_months': flat_age_months,
                  'lease_exp': lease_exp}
    
    # Creating a DataFrame with user input
    user_input = pd.DataFrame(user_data,index=[0])
    
    # Predicting resale price using the loaded model
    predict_price = model_rf.predict(user_input)
    
    # Displaying user input and predicted resale price
    st.write('User Input', user_input)
    st.write('Resale Value','S$',predict_price)
