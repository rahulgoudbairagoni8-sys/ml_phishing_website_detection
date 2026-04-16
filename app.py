"""Packages and Modules Management"""

# importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""Model Setup and Configuration"""

# defining preprocessing functions from original script
def calculate_entropy(url):
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

# loading pretrained models and assets
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('best_random_forest_grid_search_model.pkl')
        scaler = joblib.load('scaler.pkl')
        tokenizer = joblib.load('tokenizer.pkl')
        
        # building lstm model architecture manually to bypass keras version mismatch
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # explicitly building the model with expected input shape before loading weights
        lstm_model.build(input_shape=(None, 100))
        
        # loading only the weights to avoid config deserialization errors
        lstm_model.load_weights('lstm_model.keras')
        
        return rf_model, lstm_model, scaler, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# initializing models
rf_model, lstm_model, scaler, tokenizer = load_models()

"""User Interface"""

# setting up main ui
st.title("AI Phishing URL Detector")
st.markdown("Use Quick Scan to test individual URLs using the LSTM model, or Batch Processing to run tabular data through the Random Forest classifier.")

# verifying models are loaded
if rf_model and lstm_model:
    
    # creating tabs for different functionalities
    tab1, tab2 = st.tabs(["Quick Scan", "Batch Processing"])

    # handling single url prediction
    with tab1:
        st.subheader("Single URL Classification")
        st.write("Enter a raw URL to classify it using the character-level LSTM Neural Network.")
        
        user_url = st.text_input("Enter URL to analyze:")
        
        if st.button("Analyze URL"):
            if user_url:
                with st.spinner("Analyzing sequence patterns..."):
                    # preprocessing text
                    seq = tokenizer.texts_to_sequences([user_url])
                    padded_seq = pad_sequences(seq, maxlen=100)
                    
                    # predicting probability
                    pred_prob = lstm_model.predict(padded_seq)[0][0]
                    
                    st.divider()
                    if pred_prob > 0.5:
                        st.error("PHISHING DETECTED")
                        st.write(f"The model flags this URL as malicious with {pred_prob:.1%} confidence.")
                    else:
                        st.success("SAFE URL")
                        st.write(f"The model flags this URL as legitimate with {(1 - pred_prob):.1%} confidence.")
            else:
                st.warning("Please enter a URL to proceed.")

    # handling batch prediction
    with tab2:
        st.subheader("Batch Classification")
        st.write("Upload a dataset of extracted URL features to classify multiple records simultaneously.")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Applying Feature Engineering and Classifier..."):
                    try:
                        # generating engineered features if URL column exists
                        if 'URL' in df.columns:
                            df['entropy'] = df['URL'].apply(calculate_entropy)
                            df['url_length'] = df['URL'].apply(len)
                            df['special_char_count'] = df['URL'].apply(lambda x: sum([1 for c in str(x) if not c.isalnum()]))
                            keywords = ['login', 'secure', 'verify', 'bank']
                            df['keyword_flag'] = df['URL'].apply(lambda x: int(any(k in str(x).lower() for k in keywords)))
                        else:
                            st.error("The uploaded CSV must contain a 'URL' column to generate required features.")
                            st.stop()
                            
                        # dropping unnecessary and unseen columns
                        cols_to_drop = ['label', 'URL', 'Domain', 'TLD', 'Title', 'id', 'FILENAME']
                        X_features = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
                        
                        # ensuring column order matches the scaler's expected input
                        expected_features = scaler.feature_names_in_
                        
                        # checking if any required features are still missing
                        missing_features = [f for f in expected_features if f not in X_features.columns]
                        if missing_features:
                            st.error(f"Missing required features in CSV: {missing_features}")
                            st.stop()
                            
                        # aligning columns
                        X_features = X_features[expected_features]
                        
                        # scaling features
                        X_scaled = scaler.transform(X_features)
                        
                        # making predictions
                        preds = rf_model.predict(X_scaled)
                        probs = rf_model.predict_proba(X_scaled)[:, 1]
                        
                        # formatting results dataframe
                        result_df = df.copy()
                        result_df['Prediction'] = ["Phishing" if p == 1 else "Legitimate" for p in preds]
                        result_df['Phishing_Probability'] = np.round(probs, 4)
                        
                        st.write("Prediction Complete")
                        
                        # displaying metrics
                        phishing_count = (result_df['Prediction'] == "Phishing").sum()
                        st.metric("Total Phishing URLs Detected", f"{phishing_count} / {len(result_df)}")
                        
                        # showing resulting dataframe
                        st.dataframe(result_df[['URL', 'Prediction', 'Phishing_Probability']])
                        
                        # generating csv for download
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="phishing_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Prediction failed. Error: {e}")
