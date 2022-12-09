import streamlit as st 

from notebook import ml_model

def predict(input1, input2, input3, input4, input5, input6):
    input_data = [input1, input2, input3, input4, input5, input6]
    prediction = ml_model.my_method(input_data)
    return prediction

st.title("LinkedIn Predictor App")

st.text("Income:\n 1	Less than $10,000 \n 2	10 to under $20,000 \n 3	20 to under $30,000 \n 4	30 to under $40,000 \n 5	40 to under $50,000 \n 6	50 to under $75,000 \n 7	75 to under $100,000 \n 8	100 to under $150,000 \n 9	$150,000 or more? ")

input1 = st.slider(label="Enter your Income", 
          min_value=1,
          max_value=9,
          value=1)

st.text("Education (highest level of school/degree completed):\n 1	Less than high school (Grades 1-8 or no formal schooling)\n 2	High school incomplete (Grades 9-11 or Grade 12 with NO diploma)\n 3	High school graduate (Grade 12 with diploma or GED certificate)\n 4	Some college, no degree (includes some community college) \n 5	Two-year associate degree from a college or university \n 6	Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB) \n 7	Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school) \n 8	Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)")

input2 = st.slider(label="Enter your education level",
          min_value=1,
          max_value=8,
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

