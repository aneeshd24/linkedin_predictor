#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt 
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


num1 = st.slider(label="Enter your Income", 
          min_value=1,
          max_value=9,
          value=1)

num2 = st.slider(label="Enter your education level",
          min_value=1,
          max_value=9,
          value=1)

num3 = st.slider(label="Are you a parent? (0 = No & 1 = Yes)", 
          min_value=0,
          max_value=1,
          value=0)

num4 = st.slider(label="Are you married? (0 = No & 1 = Yes)", 
          min_value=0,
          max_value=1,
          value=0)

num5 = st.slider(label="What is your gender? (0 = Male & 1 = Female)", 
          min_value=0,
          max_value=1,
          value=1)

num6 = st.slider(label="What is your age?", 
          min_value=1,
          max_value=99,
          value=20)

st.write("Your numbers: ", num1, num2, num3, num4, num5, num6)

############

s = pd.read_csv("social_media_usage.csv")
s.shape
s = s.dropna()

def clean_sm(x):
    return np.where(x == 1, 1, 0)

toy_data = pd.DataFrame([[1, 90], [11, 13], [40, 21]], columns=['a', 'b'])
toy_data['a'] = toy_data['a'].apply(clean_sm)
toy_data['b'] = toy_data['b'].apply(clean_sm)
toy_data


ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
ss.insert(0, 'sm_li', clean_sm(s[['web1h']]), True) 
ss

ss['income'] = ss['income'].apply(lambda x: x if x <= 9 else np.NaN)
ss['educ2'] = ss['educ2'].apply(lambda x: x if x <= 8 else np.NaN)
ss['age'] = np.where(ss['age'] > 98, np.NaN, ss['age'])

ss['par'] = ss['par'].apply(clean_sm)
ss['marital'] = ss['marital'].apply(clean_sm)
ss['gender'] = ss['gender'].apply(clean_sm)

ss = ss.dropna()
ss

pd.crosstab(ss.educ2,ss.sm_li).plot(kind='bar')
plt.title('Exploration Graph')
plt.xlabel('Variable')
plt.ylabel('Frequency of LinkedIn Use')

pd.crosstab(ss.income,ss.sm_li).plot(kind='bar')
plt.title('Exploration Graph')
plt.xlabel('Variable')
plt.ylabel('Frequency of LinkedIn Use')

pd.crosstab(ss.par,ss.sm_li).plot(kind='bar')
plt.title('Exploration Graph')
plt.xlabel('Variable')
plt.ylabel('Frequency of LinkedIn Use')

pd.crosstab(ss.marital,ss.sm_li).plot(kind='bar')
plt.title('Exploration Graph')
plt.xlabel('Variable')
plt.ylabel('Frequency of LinkedIn Use')

pd.crosstab(ss.gender,ss.sm_li).plot(kind='bar')
plt.title('Exploration Graph')
plt.xlabel('Variable')
plt.ylabel('Frequency of LinkedIn Use')

y = ss['sm_li']
X = ss[['income', 'educ2', 'par', 'marital', 'gender', 'age']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

print(classification_report(y_test, y_pred))

newdata = pd.DataFrame({
    "income": [8, 8],
    "educ2": [7, 7],
    "par": [0, 0],
    "marital": [1, 1],
    "gender": [1, 1],
    "age": [42, 82],
})
newdata

newdata["prediction_linkedin_use"] = lr.predict(newdata)
newdata

person = [7, 7, 0, 0, 0, 22]

predicted_class = lr.predict([person])

probs = lr.predict_proba([person])

print(f"Predicted class: {predicted_class[0]}")
print(f"Probability that this person uses LinkedIn: {probs[0][1]}")
