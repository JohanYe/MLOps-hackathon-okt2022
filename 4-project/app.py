#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:33:16 2021

@author: praneeth
"""

import streamlit as st
import pandas as pd

import plotly.express as px


st.title("Menthal Health App")
st.markdown("This is a demo Streamlit app.")

@st.cache(persist=False)
def load_data():
    #df = pd.read_csv("https://datahub.io/machine-learning/iris/r/iris.csv")
    df = pd.read_csv("model_predictions.csv", sep=";")
    print(df.head())
    df['user'] = df.source.str.split("/", 1).str[1]
    df['date'] = pd.to_datetime(df.created_at)
    df['day'] = df.date.dt.date
    return(df)



def run():
    #st.subheader("Iris Data Loaded into a Pandas Dataframe.")
    st.subheader("Mental health indicator of famous people..")
    
    df = load_data()
    
    
    disp_head = st.sidebar.radio('Select DataFrame Display Option:',('Head', 'All'),index=0)
   
    #Multi-Select
    #sel_plot_cols = st.sidebar.multiselect("Select Columns For Scatter Plot",df.columns.to_list()[0:4],df.columns.to_list()[0:2])
    
    #Select Box
    #x_plot = st.sidebar.selectbox("Select X-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=0)
    #y_plot = st.sidebar.selectbox("Select Y-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=1)
    
    grouped = df.groupby([df.user, df.day, df.label]).count().reset_index()
    if disp_head=="Head":
        st.dataframe(grouped.head())
    else:
        st.dataframe(df)
    #st.table(df)
    #st.write(df)
    
    fig = px.scatter(grouped, x=grouped.day, y=grouped['id'], color="label",
                 size='id', hover_data=['label'])
    

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
