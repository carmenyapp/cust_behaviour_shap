import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import shap
import matplotlib.pyplot as plt
import numpy as np
import openai

openai.api_key =  st.secrets["mykey"]

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

# Manual cluster desc
cluster_descriptions = {
    "Cluster 0": "Value quality and premium products, seldom website visits, high income level, frequently purchases catalogs, frequently purchases fish, fruit, and sweets, seldom purchases with deals, less or no children (kid and teen) in home.",
    "Cluster 1": "Low to medium website visits, seldom purchases products, mid-income level, rarely purchases catalogs, high number of teens in home.",
    "Cluster 2": "Frequently visits website, mid-lower income level, seldom purchases products, seldom purchases in store, rarely purchases catalogs, many kids in home.",
    "Cluster 3": "Frequent to medium website visits, medium to high wine lover, frequently purchases from website and deals, medium income level, frequently purchases gold, many teens in home."
}

def marketing_text_generator():
    st.write("## Marketing Messages for Each Cluster")
    for cluster, description in cluster_descriptions.items():
        marketing_text = generate_marketing_text(cluster, description)
        st.markdown(f"### {cluster}")
        st.write(marketing_text)

def generate_marketing_text(cluster_name, cluster_description):
    prompt = f"""
    Generate a compelling marketing message for {cluster_name} based on the following customer characteristics:
    {cluster_description}
    The message should highlight relevant promotions, engagement strategies, and personalized offers.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a marketing expert specializing in personalized customer engagement."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

def customer_segmentation_and_analysis():
    # Streamlit UI Configuration
    st.set_page_config(layout="wide")
    st.title("Customer Segmentation and Cluster Analysis")
        
    # Load dataset
    df, categorical_cols = load_data()
    n_clusters = st.slider("Select Number of Cluster for Segmentation", min_value=2, max_value=6, value=3, step=1)
    
    if st.button("Segment"):
        df = apply_kprototypes(df, categorical_cols, n_clusters)
        st.session_state.segmented_df = df  # Save the segmented DataFrame
        st.session_state.clusters = sorted(df['Cluster'].unique())  
        
        X = df.drop('Cluster', axis=1)
        y = df['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write(f"F1 Score for K-Prototype: {f1:.4f}")
    
    if "segmented_df" in st.session_state and st.session_state.segmented_df is not None:
        show_classification_report = st.checkbox("Show Classification Report")
        show_feature_importance = st.checkbox("Show Feature Importance Bar Plot")
        show_shap_summary = st.checkbox("Show SHAP Summary Plot")
        show_shap_dependence = st.checkbox("Show SHAP Dependence Plot")
        show_cluster_char = st.checkbox("Show Clusters Characteristics")
    
    
        if st.button("Analyze Cluster"):
            for cluster in st.session_state.clusters:
                df = st.session_state.segmented_df.copy()
                df['binary_target'] = (df['Cluster'] == cluster).astype(int)
            
                # Features and target
                X = df.drop(columns=['Cluster', 'binary_target'])
                y = df['binary_target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
    
                if show_classification_report:
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.write("Classification Report:")
                    st.table(report_df)
                
                # SHAP analysis
                explainer = shap.TreeExplainer(model, model_output='raw')
                shap_values = explainer.shap_values(X_test)
                mean_shap = np.abs(shap_values[:,:,1]).mean(axis=0)
                
                if show_feature_importance:
                    st.subheader(f"Feature Importance - Cluster {cluster}")
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
                    plt.close()
            
                if show_shap_summary:
                    st.subheader(f"SHAP Summary Plot - Cluster {cluster}")
                    fig = plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values[:,:,1],
                        X_test,
                        feature_names=list(X.columns),
                        max_display=25,
                        plot_type="dot",
                        show=False
                    )
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
        
                if show_shap_dependence:
                    top_feature_idx = np.argmax(mean_shap)
                    top_feature_name = list(X.columns)[top_feature_idx]
                    st.subheader(f"SHAP Dependence Plot - {top_feature_name}")
                    fig_dep, ax_dep = plt.subplots(figsize=(12, 8))
                    shap.dependence_plot(
                        top_feature_idx,
                        shap_values[:,:,1],
                        X_test,
                        feature_names=list(X.columns),
                        show=False,
                        ax=ax_dep
                    )
                    st.pyplot(fig_dep)
                    plt.close()
    
            if show_cluster_char:
                st.write("## Cluster Characteristics:")  # Use markdown for a header
                # display the manual type cluster desc
                for cluster, description in cluster_descriptions.items():
                    st.markdown(f"**{cluster}:** {description}")
                
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select a feature:", ["Customer Segmentation and Cluster Analysis", "Marketing Text Generator for Clusters"])

if option == "Customer Segmentation and Cluster Analysis":
    customer_segmentation_and_analysis()
elif option == "Marketing Text Generator for Clusters":
    marketing_text_generator()
