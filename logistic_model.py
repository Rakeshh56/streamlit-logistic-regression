


import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pclass = st.sidebar.selectbox('Pclass',('1','2','3'))
    Sex= st.sidebar.selectbox('Gender',('1','0'))
    Age = st.sidebar.number_input("Insert the Age")
    Fare=st.sidebar.number_input("Insert Fare")
    Embarked=st.sidebar.selectbox('Embarked',('0','1','2'))
    data = {
            'Pclass':Pclass,
             'Sex':Sex,
            'Age':Age,
            'Fare':Fare,
            'Embarked':Embarked}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model=load(open('file_name.sav','rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)


