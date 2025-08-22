
import streamlit as st
import pickle
import numpy as np

# Load all saved models and data
similarity_df = pickle.load(open("similarity_matrix.pkl", "rb"))
desc_to_code = pickle.load(open("desc_to_code.pkl", "rb"))
code_to_desc = pickle.load(open("code_to_desc.pkl", "rb"))
kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


#  Helper Functions


# 1. Product Recommendations
def get_similar_products(product_name, top_n=5):
    try:
        product_code = desc_to_code[product_name]
        sim_scores = similarity_df[product_code].sort_values(ascending=False)[1:top_n+1]
        return [code_to_desc[code] for code in sim_scores.index if code in code_to_desc]
    except KeyError:
        return ["Product not found."]

# 2. Customer Segmentation
def label_cluster_from_id(cluster_id):
    if cluster_id == 2:
        return 'High-Value'
    elif cluster_id == 3:
        return 'Loyal'
    elif cluster_id == 0:
        return 'Occasional'
    else:
        return 'At-Risk'

def predict_customer_segment(recency, frequency, monetary):
    input_scaled = scaler.transform(np.array([[recency, frequency, monetary]]))
    cluster_id = kmeans_model.predict(input_scaled)[0]
    label = label_cluster_from_id(cluster_id)
    return cluster_id, label


#  Streamlit App


st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("🛒 Shopper Spectrum")

# Create tabs
tab1, tab2 = st.tabs(["Product Recommender", "Customer Segmentation"])


#  TAB 1: Recommender

with tab1:
    st.header("Product Recommender")
    st.markdown("Select a product to get similar product recommendations:")

    product_names = sorted(desc_to_code.keys())
    product_input = st.selectbox("Choose a product:", product_names)

    if st.button("Get Recommendations"):
        recommendations = get_similar_products(product_input)
        st.markdown("### 🧾 Recommended Products:")
        for i, item in enumerate(recommendations, start=1):
            st.markdown(f"{i}. {item}")


# 📊 TAB 2: Segmentation

with tab2:
    st.header("Customer Segmentation")
    st.markdown("Enter RFM values to predict the customer segment.")

    r = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    f = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    m = st.number_input("Monetary (total spend)", min_value=0.0, step=10.0)

    if st.button("Predict Segment"):
        cluster_id, segment = predict_customer_segment(r, f, m)
        st.success(f"Predicted Segment: **{segment}** (Cluster {cluster_id})")




# 1. Product Recommendations
def get_similar_products(product_name, top_n=5):
    try:
        product_code = desc_to_code[product_name]
        sim_scores = similarity_df[product_code].sort_values(ascending=False)[1:top_n+1]
        return [code_to_desc[code] for code in sim_scores.index if code in code_to_desc]
    except KeyError:
        return ["Product not found."]

# 2. Customer Segmentation
def label_cluster_from_id(cluster_id):
    if cluster_id == 2:
        return 'High-Value'
    elif cluster_id == 3:
        return 'Loyal'
    elif cluster_id == 0:
        return 'Occasional'
    else:
        return 'At-Risk'

def predict_customer_segment(recency, frequency, monetary):
    input_scaled = scaler.transform(np.array([[recency, frequency, monetary]]))
    cluster_id = kmeans_model.predict(input_scaled)[0]
    label = label_cluster_from_id(cluster_id)
    return cluster_id, label



