import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.title('Resale Price Prediction for Singapore Flats')
st.write("Welcome to the Singapore Resale Price Predictor! This application is designed to help you estimate "
    "the resale price of a flat based on historical transaction data. Simply input the details of the flat, "
    "and let the machine learning model provide you with an estimated resale price.")

st.sidebar.header('Enter Values')

# Get the User Input

def user_input():
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

    user_data = { 'town' : town,
                  'flat_type' : flat_type,
                  'storey_range': Storey,
                  'floor_area_sqm': floor_area,
                  'flat_model': flat_model,
                  'flat_age_months': flat_age_months,
                  'lease_exp': lease_exp}

    features = pd.DataFrame(user_data,index=[0])
    return features

input_df = user_input()
user_submit = st.sidebar.button('Submit')

# applying the model to make prediction
if user_submit == True:
    # Read the random forest regression ML model
    house_price = pickle.load(open('singapore_house_resale_price_prediction.pkl', 'rb'))
    st.write('User Input', input_df)
    predict_price = house_price.predict(input_df)
    st.write('Resale Value','S$',predict_price)

