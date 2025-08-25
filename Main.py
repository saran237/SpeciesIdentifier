#This is a sample python script.


import streamlit as st
import pandas as pd
from os import path
import pickle


st.title("Flower Species Predictor")
df_iris = pd.read_csv(path.join("Data",'iris.csv'))
# st.write(df_iris)
# st.scatter_chart(df_iris[['sepal_length','sepal_width']])

petal_length=st.number_input("Please choose a petal_length",placeholder="Enter a value ranges between 1.0 and 6.9",min_value=1.000000	,max_value=6.900000,value=None)
petal_width=st.number_input("Please choose a petal_width",placeholder="Enter a value ranges between 0.1 and 2.5",min_value=0.1,max_value=2.5,value=None)
sepal_length=st.number_input("Please choose a sepal_length",placeholder="Enter a value ranges between 4.3 and 7.9",min_value=4.300000,max_value=7.900000,value=None)
sepal_width=st.number_input("Please choose a sepal_width",placeholder="Enter a value ranges between 2.0 and 4.4",min_value=2.000000,max_value=4.400000,value=None)

#prepare the dataframe for prediction
user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns=['sepal_length','sepal_width','petal_length','petal_width'])

#using the pkl file creating a ml model named 'iris_predictor'
model_path = path.join("Model","iris_model.pkl")
with open(model_path, 'rb') as file:
    iris_predictor = pickle.load(file)
st.write(user_input)


dict_species={0:'setosa',1:'versicolor',2:'virginica'}


if st.button("Predict Species"):
    if((petal_width==None) or (sepal_length==None) or (sepal_length==None) or (sepal_width==None)):
        # will be executed when any of the values is not entered properly
        st.write("Please fill all values")
    else:
        #prediction can be done here
        predicted_species = iris_predictor.predict(user_input)
        #predicted_species[0] will give the value in the data frame
        #we use that value to find the corresponding species from the dictionary 'dict_species'
        st.write("The Species is ",dict_species[predicted_species[0]])