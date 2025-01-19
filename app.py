import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score
from kmodes.kprototypes import KPrototypes
import shap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.title("Customer Segmentation and Cluster Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("marketing_campaign.csv", sep='\t')
    # Preprocess dataset
    df = df.drop(['ID','Year_Birth','Z_CostContact', 'Z_Revenue'], axis=1)
    df['Income'] = df['Income'].fillna(-1.0)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    df['Customer_Tenure'] = df['Dt_Customer'].apply(
        lambda x: (datetime.now() - x).days if pd.notnull(x) else -1
    )
    df = df.drop(['Dt_Customer'], axis=1)
    
    # Label encoding categorical columns
    categorical_cols = ['Education', 'Marital_Status',
                        'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
                        'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Standard scaling for numeric columns
    numeric_cols = ['Income', 'Recency', 'NumWebPurchases', 'NumStorePurchases',
                    'NumCatalogPurchases', 'NumDealsPurchases', 'MntWines',
                    'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                    'MntSweetProducts', 'MntGoldProds', 'Customer_Tenure']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, categorical_cols
    
def apply_kprototypes(df, categorical_cols, n_clusters):
    # K-Prototypes clustering
    kproto = KPrototypes(n_clusters=n_clusters, init='Huang', random_state=42)
    categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]
    clusters = kproto.fit_predict(df, categorical=categorical_indices)
    df['Cluster'] = clusters
    return df
    
# Load dataset
df, categorical_cols = load_data()
n_clusters = st.slider("Select Number of Cluster for Segmentation", min_value=2, max_value=6, value=3, step=1)
df = apply_kprototypes(df, categorical_cols, n_clusters)

# X = df.drop('Cluster', axis=1)
# y = df['Cluster']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a classifier
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train, y_train)

# # Predict on the test set
# y_pred = clf.predict(X_test)

# # Evaluate the F1 score
# f1 = f1_score(y_test, y_pred, average='weighted')
# st.write(f"F1 Score: {f1:.4f}")

clusters = sorted(df['Cluster'].unique())
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
    explainer = shap.TreeExplainer(model, model_output='raw')
    shap_values = explainer.shap_values(X_test)
    
    # 1. Feature Importance Bar Plot
    st.subheader(f"Feature Importance - Cluster {selected_cluster}")
    mean_shap = np.abs(shap_values[1]).mean(axis=0)  
    feature_importance = pd.DataFrame(mean_shap, index=list(X.columns), columns=['SHAP Value'])
    feature_importance = feature_importance.sort_values('SHAP Value', ascending=True)

    # Bar plot for feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_importance)), feature_importance['SHAP Value'])
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance.index)
    ax.set_xlabel('mean(|SHAP value|)')
    ax.set_title(f'Feature Importance - Cluster {selected_cluster}')
    st.pyplot(fig)

    # 2. SHAP Summary Plot
    st.subheader(f"SHAP Summary Plot - Cluster {selected_cluster}")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values[1],
        X_test,
        feature_names=list(X.columns),
        max_display=25,
        show=False,
    )
    st.pyplot(plt.gcf())
    plt.close()

    top_feature_idx = np.argmax(mean_shap)
    top_feature_name = list(X.columns)[top_feature_idx]
    
    st.subheader(f"SHAP Dependence Plot - {top_feature_name}")
    fig_dep, ax_dep = plt.subplots(figsize=(12, 8))
    shap.dependence_plot(
        top_feature_idx,
        shap_values[1],
        X_test,
        feature_names=list(X.columns),
        show=False,
        ax=ax_dep
    )
    st.pyplot(fig_dep)
    plt.close()

