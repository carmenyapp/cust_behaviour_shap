import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.title("Cluster-wise Analysis")

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

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("Classification Report:")
    st.table(report_df)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 1. Feature Importance Bar Plot
    st.subheader(f"Feature Importance - Cluster {selected_cluster}")
    mean_shap = np.abs(shap_values[1]).mean(axis=0)  
    feature_importance = pd.DataFrame(mean_shap, index=list(X.columns), columns=['SHAP Value'])
    feature_importance = feature_importance.sort_values('SHAP Value', ascending=True)

    # Bar plot for feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(feature_importance)), feature_importance['SHAP Value'])
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance.index)
    ax.set_xlabel('mean(|SHAP value|)')
    ax.set_title(f'Feature Importance Plot - Cluster {selected_cluster}')
    st.pyplot(fig)

    # 2. SHAP Summary Plot
    st.subheader("SHAP Summary Plot")
    shap.summary_plot(
        shap_values[1],
        X_test,
        feature_names=list(X.columns),
        max_display=25,
        show=False
    )
    st.pyplot(plt.gcf())
