import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

#Function blocks

def Selling_Price():
    st.write('Fill the appropirate details to predict the Copper Selling Price')

    #Creating dictionary for mapping string to integer
    status_dic={'Won':7, 'Lost':1, 'Not lost for AM':2, 'Revised':5, 'To be approved':6, 'Draft':0, 'Offered':4, 'Offerable':3, 'Wonderful':8}
    item_type_dic={'W':5, 'S':3, 'PL':2, 'Others':1, 'WI':6, 'IPL':0, 'SLAWR':4}

    #dropdown for categorical variables
    country_options=[25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
    status_options=['Won', 'Lost', 'Not lost for AM', 'Revised', 'To be approved', 'Draft', 'Offered', 'Offerable', 'Wonderful']
    item_type_options=['W', 'S', 'PL', 'WI', 'IPL', 'SLAWR', 'Others']
    application_options=[2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    product_ref_options=[611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407, 164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

    with st.form('Selling_price'):

        col1, col2, col3=st.columns([5,1,5])

        with col1:
            quantity=st.number_input(label='**Quantity Tons (Range 392-1621)**', min_value=392.0, max_value=1621.0)
            country=st.selectbox(label='**Country Code**', options=country_options)
            item_type=st.selectbox(label='**Item Type**', options=item_type_options)
            thickness=st.number_input(label='**Thickness (Range 0.18-27.0)**', min_value=0.18, max_value=27.0)
            prod_ref=st.selectbox(label='**Product Reference**', options=product_ref_options)


        with col3:
            customer=st.number_input(label='**Customer ID**', min_value=30147616.0, max_value=2147483647.0)
            status=st.selectbox(label='**Status**', options=status_options)
            application=st.selectbox(label='**Application code**', options=application_options)
            width=st.number_input(label='**Width (Range 700-1980)**', min_value=700.0, max_value=1980.0)
            st.write('')
            st.write('')

            Button=st.form_submit_button(label='**:red[Predict!]**')

    col1, col2=st.columns(2)
    with col2:
        st.caption(body="All dimentions in 'mm' ")

    #process user input and make prediction
    if Button:
        #load the saved model
        with open('C:/Capstone files/Industrial_Copper_Modelling/Regression_model.pkl', 'rb') as file1:
            model_1=pickle.load(file1)

        #convert the status and item_type to their correstponding integers
        status_int=status_dic.get(status)
        item_int=item_type_dic.get(item_type)

        #Make prediction
        data1=np.array([[np.log(float(quantity)), customer, country, status_int, item_int, application, np.log(float(thickness)), width, prod_ref]])
        tar_predict=model_1.predict(data1)
        selling_price=np.exp(tar_predict[0]) # Apply exponential to get back original scale
        selling_price=round(selling_price, 2)

        st.header(f':green[Predicted Selling Price : ${selling_price}]')



def Predict_Status():
    st.write('Fill the appropirate details to find the Predicted Status')

    #Creating dictionary for mapping string to integer
    item_type_dic={'W':5, 'S':3, 'PL':2, 'Others':1, 'WI':6, 'IPL':0, 'SLAWR':4}

    #dropdown for categorical variables
    country_options=[25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
    item_type_options=['W', 'S', 'PL', 'WI', 'IPL', 'SLAWR', 'Others']
    application_options=[2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    product_ref_options=[611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407, 164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]


    with st.form('Status'):

        col1, col2, col3=st.columns([5, 1, 5])
        with col1:
            quantity=st.number_input(label='**Quantity Tons (Range 392-1621)**', min_value=392.0, max_value=1621.0)
            country=st.selectbox(label='**Country Code**', options=country_options)
            item_type=st.selectbox(label='**Item Type**', options=item_type_options)
            thickness=st.number_input(label='**Thickness (Range 0.18-27.0)**', min_value=0.18, max_value=27.0)
            prod_ref=st.selectbox(label='**Product Reference**', options=product_ref_options)

        with col3:

            customer=st.number_input(label='**Customer ID**', min_value=30147616.0, max_value=2147483647.0)
            sell_price=st.number_input(label='**Selling Price (Range 390 - 1620)**', min_value=390, max_value=1620)
            application=st.selectbox(label='**Application code**', options=application_options)
            width=st.number_input(label='**Width (Range 700-1980)**', min_value=700.0, max_value=1980.0)
            st.write('')
            st.write('')

            Button=st.form_submit_button(label='**:red[Predict!]**')

    col1, col2=st.columns(2)
    with col2:
        st.caption(body="All dimentions in 'mm' ")

    if Button:
        #load the saved model
        with open('C:/Capstone files/Industrial_Copper_Modelling/Classification_model.pkl', 'rb') as file2:
            model_2=pickle.load(file2)     

        #convert the status and item_type to their correstponding integers
        item_int=item_type_dic.get(item_type)   

        #Make prediction
        data2=np.array([[np.log(float(quantity)), customer, country, item_int, application, np.log(float(thickness)), width, prod_ref, np.log(float(sell_price))]])
        tar_predict=model_2.predict(data2)  

        if tar_predict[0]==7:
            st.header(':green[Predicted Status : Won]')
        else:
            st.header(':red[Predicted Status is : Lose]')   


def home():
    st.subheader(':violet[Overview]')
    st.markdown('**The Industrial Copper Modeling project aims to address the challenges faced by the copper industry in accurately predicting sales prices and classifying leads. This project focuses on developing and analysis the multiple machine learning models to enhance decision-making processes related to pricing and lead conversion.**') 
    st.markdown('**:violet[Domine :] Manufacturing**')
    st.subheader(':violet[Features and Goals ]') 
    st.markdown("""
                - **:green[Data Exploration] : Analyze the dataset for skewness and outliers.**
                - **:green[Data Transformation] : Clean and preprocess the data for modeling.**
                - **:green[Regression Modeling] : Predict the `Selling_Price` using machine learning models.**
                - **:green[Classification Modeling] : Classify leads as `WON` or `LOST`.**
                - **:green[Interactive Dashboard] : Use Streamlit for dynamic data exploration and predictions.**

                """)
    
    st.write('')
    st.markdown("**:violet[Technologies Used :] Pandas, Numpy, Scikit-learn, Seaborn, Matplotlib and Streamlit**")

    st.write('')
    st.markdown("**:blue[Github link - ] https://github.com/vigneshwds/Industrial_Copper_Modeling.git**")

                

#Streamlit

st.title(':green[Industrial Copper Modeling]')

with st.sidebar:
    user=option_menu('Menu', ['Home', 'Prediction', 'About'], icons=["house", "graph-up", "book"], menu_icon="cast")

if user=='Home':
    home()
    st.image(Image.open('C:\Capstone files\Industrial_Copper_Modelling\Footer.png'), width=640)
elif user=='Prediction':
    option=option_menu('', ['Selling Price Prediction', 'Transaction Status Prediction'], orientation='horizontal', icons=['check', 'check'])
    if option=='Selling Price Prediction':
        st.subheader(':blue[Price Prediction]')
        Selling_Price()
    elif option=='Transaction Status Prediction':
        st.subheader(':blue[Status Prediction]')
        Predict_Status()
elif user=='About':
    st.subheader(':green[Methodology]')
    st.markdown('##### :violet[1. Data Exploration and Preprocessing]')
    st.markdown("""
                - **Conduct exploratory data analysis (EDA) to uncover patterns, correlations, and anomalies.**
                - **Address skewness and noise through techniques such as log transformation and data normalization.**
                - **Detect and handle outliers to ensure data quality and model reliability.**
                """)
    st.markdown('##### :violet[2. Regression Model Development]')
    st.markdown("""
                - **Implement a machine learning regression model using algorithms robust to skewed and noisy data, such as Random Forest or Gradient Boost.**
                - **Optimize model parameters through techniques like GridSearchCV to achieve optimal performance.**
                """)
    st.markdown('##### :violet[3. Classification Model Development]')
    st.markdown("""
                - **Develop a lead classification model using algorithm Random Forset Classifier.**
                - **Focus on the STATUS variable, considering WON as success and LOST as failure, and remove irrelevant data points.**
                """)
    st.markdown('##### :violet[4. Streamlit Dashboard Implementation]')
    st.markdown("""
                - **Design an intuitive Streamlit interface to facilitate user interaction and data input.**
                - **Integrate predictive models into the dashboard to provide real-time insights and predictions.**
                """)
    
    st.subheader(':green[Conculstion]')
    st.markdown("**The Industrial Copper Modeling project aims to revolutionize the copper industry's approach to pricing and lead management through advanced machine learning techniques. By addressing data challenges and providing actionable insights, this project will enable more informed and efficient decision-making, ultimately leading to increased profitability and customer satisfaction.**")
