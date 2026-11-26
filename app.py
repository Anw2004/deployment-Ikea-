# Streamlit application

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("IKEA_product_catalog-2.csv")

st.title("IKEA Product Clustering Application")
st.plotly_chart(px.histogram(df, x='product_rating', marginal='box', nbins=30, title='Product Rating Distribution'))



clustering = joblib.load('ikea_pipeline.pkl')


def predict_cluster(price, product_rating, product_rating_count):
    input_data = pd.DataFrame({
        'price': [price],
        'product_rating': [product_rating],
        'product_rating_count': [product_rating_count]
    })

    prediction = clustering.predict(input_data)

    return prediction[0]


def main():
    st.title('IKEA Product Clustering')

    html_temp = """
    <div style="background-color:#0058A3;padding:10px">
        <h2 style="color:#FFDA1A;text-align:center;">
            IKEA Product Segmentation App
        </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    price = st.number_input("Product Price", min_value=0.0, value=150.0)

    product_rating = st.number_input("Product Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

    product_rating_count = st.number_input("Number of Reviews", min_value=1, value=100)

    if st.button("Predict Cluster"):
        result = predict_cluster(price, product_rating, product_rating_count)
        st.success(f"Predicted Cluster: {result}")


if __name__ == '__main__':
    main()
