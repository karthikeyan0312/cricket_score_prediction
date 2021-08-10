
import streamlit as st
import pandas as pd
@st.cache
def load_data():
    df=pd.read_excel(r"/app/cricket_score_prediction/ipl.csv")
    return df
dataset = load_data()
X = dataset.iloc[:,[7,8,9,12,13]].values #Input features
y = dataset.iloc[:, 14].values #Label

import numpy as np



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000,max_features=None,random_state=12)
model.fit(X_train,y_train)


#print("Random Regression-Model Accuracy ",model.score(X_test,y_test)*100)


#new_prediction = model.predict(sc.transform(np.array([[100,0,13,50,50]])))
#print("Prediction score:" , new_prediction)

placeholder = st.empty()

input1 = st.number_input("Runs")
input2 = st.number_input("Wickets")
input3 = st.number_input("Overs")
input4 = st.number_input("Striker")
input5 = st.number_input("Non-Striker")

submit = st.button(" Predict ")
if submit:
    new_prediction = model.predict(sc.transform(np.array([[input1,input2,input3,input4,input5]])))
    st.write("Predicted score : "+str(new_prediction[0]))
