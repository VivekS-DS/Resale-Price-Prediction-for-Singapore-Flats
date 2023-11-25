import pickle
import streamlit as st
# Import Pandas for data manipulation
import pandas as pd
# Import NumPy for numerical operations
import numpy as np
# For composing transformers for different data types
from sklearn.compose import make_column_transformer
# For building a machine learning pipeline
from sklearn.pipeline import Pipeline
# For preprocessing data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# For using a Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
# For using a Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split
# For evaluating the model performance
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
# Import warnings module to suppress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings during execution for cleaner output

st.title('Resale Price Prediction for Singapore Flats')
st.write("Welcome to the Singapore Flat Resale Price Predictor! This application is designed to help you estimate "
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
user_submit = st.sidebar.button('Submit')
# ML Model
def ml_model():
    house_ml = pd.read_csv('https://raw.githubusercontent.com/VivekS-DS/Resale-Price-Prediction-for-Singapore-Flats/main/sing_house_price_cleaned.csv')
    X = house_ml.drop('resale_price', axis=1)
    Y = house_ml['resale_price']
    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # ColumnTransformer
    ct = make_column_transformer((StandardScaler(), ['floor_area_sqm', 'flat_age_months', 'lease_exp']),
                                     (OneHotEncoder(), ['town', 'flat_type', 'storey_range', 'flat_model']),
                                     remainder="drop"  # all other columns in X will be dropped.
                                     )
    # Create a pipeline with the ColumnTransformer and a model
    model_rf = Pipeline(steps=[('processor', ct),
                                   ('regressor', RandomForestRegressor())])
    model_rf.fit(x_train, y_train)
    rf_train_pred = model_rf.predict(x_train)
    rf_test_pred = model_rf.predict(x_test)
    return model_rf



model = ml_model()
input_df = user_input()

# applying the model to make prediction
if user_submit == True:
    st.write('User Input', input_df)
    predict_price = model_rf.predict(input_df)
    st.write('Resale Value','S$',predict_price)

