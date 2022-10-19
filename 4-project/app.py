#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:33:16 2021

@author: praneeth
"""

import streamlit as st
import pandas as pd
from test import predict
import plotly.express as px


st.title("Menthal Health App")
st.markdown("This is a demo Streamlit app.")

#@st.cache(persist=False)
def load_data():
    #df = pd.read_csv("https://datahub.io/machine-learning/iris/r/iris.csv")
    df = pd.read_csv("model_predictions.csv", sep=";")
    print(df.head())
    df['prob'] = df.prob.astype(float)
    df['user'] = df.source.str.split("/", 1).str[1]
    df['date'] = pd.to_datetime(df.created_at)
    df['day'] = df.date.dt.date
    return(df)

@st.cache(persist=True)
def make_predict_request(twitter_handle):
    print(twitter_handle)
    if len(twitter_handle) == 0:
        return
    twitter_accounts = twitter_handle.split(",")
    print(twitter_accounts)
    predict(twitter_accounts)
    load_data()


def run():
    #st.subheader("Iris Data Loaded into a Pandas Dataframe.")
    st.subheader("Mental health indicator of famous people..")
    
    twitter_handle = st.text_input(label='Twitter')
    make_predict_request(twitter_handle)
    df = load_data()
    
    plot_button = st.sidebar.radio('Plot:',('All','Depression', 'Anxiety','BPD','Autism','Bipolar','Mentalhealth','Schizophrenia'),index=0)
    
    grouped = df.groupby([df.user, df.day, df.label]).count().reset_index()
    grouped['Number of tweets'] = grouped.id

    if plot_button !='All':
        fig = px.scatter(grouped[grouped.label==plot_button.lower()], x="day", y='Number of tweets', color="user",
                 size='id', hover_data=['user'], symbol='user', title=plot_button.capitalize())
    else:
        fig = px.scatter(grouped, x=grouped.day, y=grouped['Number of tweets'], color="label",
                 size='id', hover_data=['user'], symbol='user', title=plot_button.capitalize())

    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # Display a static table
    #st.table(df)
    if plot_button != 'All':
        #st.dataframe(df[df.label==plot_button.lower()].sort_values(df[df.label==plot_button.lower()].prob).head(3))
        #st.dataframe(df[df.label==plot_button.lower()].sort_values(by='prob',ascending=False)[['label','user','cleaned_text']].head(3))
        st.table(df[df.label==plot_button.lower()].sort_values(by='prob',ascending=False)[['prob','label','user','cleaned_text']].head(3))
    else:
        #st.dataframe(df.sort_values(by='prob',ascending=False)[['prob','label','user','cleaned_text']].head(3))
        st.table(df.sort_values(by='prob',ascending=False)[['prob','label','user','cleaned_text']].head(3))
    

    #Scatter Plot
    #fig = px.scatter(df, x=df["tweets"], y=df["sepalwidth"], color="class",
    #             size='petallength', hover_data=['petalwidth'])
    
    fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
   
    st.write("\n")
    st.subheader("Scatter Plot")
    st.plotly_chart(fig, use_container_width=True)
    
    
    #Add images
    #images = ["<image_url>"]
    #st.image(images, width=600,use_container_width=True, caption=["Iris Flower"])
   
    
   
    
   
if __name__ == '__main__':
    run()    
