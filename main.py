# Main pipeline to execute all steps
# Combine preprocessing, feature extraction, clustering, classification\
    
from preprocessing import load_and_preprocess
from lstm_feature_extraction import build_lstm
from bbo_fcm import bbo_fcm_clustering
from elm_classifier import ELMClassifier

import numpy as np

def run_pipeline(csv_path):
    X_scaled, y = load_and_preprocess(csv_path)

    # Reshape for LSTM
    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # LSTM model
    lstm_model = build_lstm((1, X_scaled.shape[1]))
    lstm_model.fit(X_seq, y, epochs=5, batch_size=16, verbose=1)
    features = lstm_model.predict(X_seq)

    # Clustering using BBO-FCM
    clusters = bbo_fcm_clustering(features, n_clusters=3)
    print("Cluster labels:", clusters)

    # ELM classification
    elm = ELMClassifier(input_dim=features.shape[1], hidden_dim=50)
    elm.fit(features, y)
    print("ELM Accuracy:", elm.score(features, y))

if __name__ == "__main__":
    run_pipeline("healthcare_dataset.csv")
