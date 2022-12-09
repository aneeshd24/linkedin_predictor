import streamlit as st 

from notebook import ml_model

def predict(input1, input2, input3, input4, input5, input6):
    input_data = [input1, input2, input3, input4, input5, input6]
    prediction = ml_model.my_method(input_data)
    return prediction

st.title("LinkedIn Predictor App")

input1 = st.slider(label="Enter your Income", 
          min_value=1,
          max_value=9,
          value=1)

input2 = st.slider(label="Enter your education level",
          min_value=1,
          max_value=9,
          value=1)

input3 = st.slider(label="Are you a parent? (0 = No & 1 = Yes)", 
          min_value=0,
          max_value=1,
          value=0)

input4 = st.slider(label="Are you married? (0 = No & 1 = Yes)", 
          min_value=0,
          max_value=1,
          value=0)

input5 = st.slider(label="What is your gender? (0 = Male & 1 = Female)", 
          min_value=0,
          max_value=1,
          value=0)

input6 = st.slider(label="What is your age?", 
          min_value=1,
          max_value=99,
          value=20)


if st.button("Predict"):
    print(input1, input2, input3, input4, input5, input6)
    prediction = predict(input1, input2, input3, input4, input5, input6)
    st.write("Representation for LinkedIn User: 0 = No & 1 = Yes")
    st.write("Prediction:", prediction[0])
    st.write("Probability you are a LinkedIn User ", prediction[1])

