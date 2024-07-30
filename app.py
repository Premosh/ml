import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model, label encoders, and scaler
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Define the columns based on the dataset
columns = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
    'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
]

# Create a function for prediction
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    for column, le in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = le.transform(input_df[column])
    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)
    return prediction

# Streamlit app
st.title('Lung Cancer Prediction Web App')
st.write('Input your data for prediction:')

input_data = {}
input_data['GENDER'] = st.selectbox('Gender', ['M', 'F'])
input_data['AGE'] = st.number_input('Age', min_value=0, max_value=120, value=30, step=1)
input_data['SMOKING'] = st.selectbox('Smoking', ['Yes', 'No'])
input_data['YELLOW_FINGERS'] = st.selectbox('Yellow Fingers', ['Yes', 'No'])
input_data['ANXIETY'] = st.selectbox('Anxiety', ['Yes', 'No'])
input_data['PEER_PRESSURE'] = st.selectbox('Peer Pressure', ['Yes', 'No'])
input_data['CHRONIC_DISEASE'] = st.selectbox('Chronic Disease', ['Yes', 'No'])
input_data['FATIGUE'] = st.selectbox('Fatigue', ['Yes', 'No'])
input_data['ALLERGY'] = st.selectbox('Allergy', ['Yes', 'No'])
input_data['WHEEZING'] = st.selectbox('Wheezing', ['Yes', 'No'])
input_data['ALCOHOL_CONSUMING'] = st.selectbox('Alcohol Consuming', ['Yes', 'No'])
input_data['COUGHING'] = st.selectbox('Coughing', ['Yes', 'No'])
input_data['SHORTNESS_OF_BREATH'] = st.selectbox('Shortness of Breath', ['Yes', 'No'])
input_data['SWALLOWING_DIFFICULTY'] = st.selectbox('Swallowing Difficulty', ['Yes', 'No'])
input_data['CHEST_PAIN'] = st.selectbox('Chest Pain', ['Yes', 'No'])

if st.button('Predict'):
    prediction = predict(input_data)
    st.write(f"Prediction: {'YES' if prediction[0] else 'NO'}")
