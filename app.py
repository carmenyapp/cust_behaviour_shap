import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.title("Cluster-wise Binary Classification and SHAP Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("cluster_marketing_campaign.csv")

# Load dataset
df = load_data()

# Sidebar: Cluster selection
clusters = df['Cluster'].unique()
selected_cluster = st.selectbox("Select a Cluster for Analysis", clusters)

if st.button("Analyze Cluster"):
    # Binary target for the selected cluster
    st.write(f"Analyzing Cluster {selected_cluster}")
    df['binary_target'] = (df['Cluster'] == selected_cluster).astype(int)

    # Features and target
    X = df.drop(columns=['Cluster', 'binary_target'])
    y = df['binary_target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and classification report
    y_pred = model.predict(X_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)

    # SHAP summary plot (Beeswarm)
    st.subheader(f"SHAP Impact on Model Output - Cluster {selected_cluster}")
    fig, ax = plt.subplots(figsize=(14, 8))
    shap.summary_plot(shap_values[1], X_test, show=False)
    st.pyplot(fig)

    # Feature importance plot
    st.subheader(f"Feature Importance - Cluster {selected_cluster}")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(feature_importance['feature'], feature_importance['importance'])
    ax.set_title(f'Random Forest Feature Importance - Cluster {selected_cluster}')
    ax.set_xlabel('Feature Importance')
    st.pyplot(fig)
