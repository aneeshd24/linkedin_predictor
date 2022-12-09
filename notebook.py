#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class ml_model():
    def clean_sm(x):
        return np.where(x == 1, 1, 0)
    
    def my_method(new_list):

        s = pd.read_csv("social_media_usage.csv")
        s.shape
        s = s.dropna()

        toy_data = pd.DataFrame([[1, 90], [11, 13], [40, 21]], columns=['a', 'b'])
        toy_data['a'] = toy_data['a'].apply(ml_model.clean_sm)
        toy_data['b'] = toy_data['b'].apply(ml_model.clean_sm)
        toy_data


        ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
        ss.insert(0, 'sm_li', ml_model.clean_sm(s[['web1h']]), True) 
        ss

        ss['income'] = ss['income'].apply(lambda x: x if x <= 9 else np.NaN)
        ss['educ2'] = ss['educ2'].apply(lambda x: x if x <= 8 else np.NaN)
        ss['age'] = np.where(ss['age'] > 98, np.NaN, ss['age'])

        ss['par'] = ss['par'].apply(ml_model.clean_sm)
        ss['marital'] = ss['marital'].apply(ml_model.clean_sm)
        ss['gender'] = ss['gender'].apply(ml_model.clean_sm)

        ss = ss.dropna()
        ss

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
        newPF = pd.DataFrame({
            "income": [new_list[0],new_list[0]],
            "educ2": [new_list[1],new_list[1]],
            "par": [new_list[2],new_list[2]],
            "marital": [new_list[3],new_list[3]],
            "gender": [new_list[4], new_list[4]],
            "age": [new_list[5],new_list[5]],
        })

        predicted_class = lr.predict(newPF)
        probs = lr.predict_proba(newPF)

        return [predicted_class[0], probs[0][1]]
