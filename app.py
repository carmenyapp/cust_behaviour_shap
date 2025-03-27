import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, silhouette_score
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
    
def generate_cluster_description_from_shap(shap_values, X_test, feature_names, cluster_id, segmented_df):
    print("Shape of shap_values:", shap_values.shape)
    print("Shape of X_test:", X_test.shape)
    print("Number of feature names:", len(feature_names))
    
    cluster_indices = segmented_df[segmented_df['Cluster'] == cluster_id].index

    print("Cluster ID:", cluster_id)
    print("Cluster indices:", cluster_indices)
    print("Max index in cluster_indices:", max(cluster_indices) if len(cluster_indices) > 0 else "No indices")
    
    valid_indices = [idx for idx in cluster_indices if idx < shap_values.shape[0]]
    if not valid_indices:
        return "No valid data for this cluster"
    cluster_shap_values = shap_values[cluster_indices]
    avg_shap = np.mean(cluster_shap_values, axis=0)
    top_positive_features_indices = np.argsort(avg_shap)[::-1][:5]
    top_negative_features_indices = np.argsort(avg_shap)[:5]

    positive_features = [feature_names[i] for i in top_positive_features_indices]
    negative_features = [feature_names[i] for i in top_negative_features_indices]

    prompt = f"""
    Analyze the following customer segment based on the most influential features identified by SHAP analysis.
    Highlight the key characteristics of this segment.

    Features that positively influence the cluster membership (high values in these features): {', '.join(positive_features)}
    Features that negatively influence the cluster membership (low values in these features): {', '.join(negative_features)}

    Based on these characteristics, suggest potential marketing angles or product recommendations that might resonate with this segment.

    Provide a concise description (around 60 words) of this customer segment, followed by the marketing suggestions.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in interpreting SHAP analysis for customer segmentation and generating marketing insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Error generating cluster description from SHAP: {e}")
        return "Description could not be generated."

def generate_ai_message(cluster_description, user_name, user_case):
    prompt = f"""
    Generate a personalized marketing message based on the following:

    Customer Characteristics and Marketing Suggestions: {cluster_description}
    User Requirement: {user_case}
    """
    if user_name:
        prompt = f"Generate a personalized message for {user_name} based on the following:\n\n" + prompt

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a marketing expert specializing in personalized customer engagement."},
                {"role": "user", "content": prompt}
            ]
        )
        marketing_text = response["choices"][0]["message"]["content"].strip()
        return marketing_text
    except Exception as e:
        st.error(f"Error generating AI message: {e}")
        return "Message could not be generated."

# Streamlit UI Configuration
st.title("Customer Segmentation, Analysis, and Message Generation")

if 'n_clusters_determined' not in st.session_state:
    st.session_state['n_clusters_determined'] = False
if 'n_clusters_value' not in st.session_state:
    st.session_state['n_clusters_value'] = 3 

# Load dataset
df, categorical_cols = load_data()

# Automated Cluster Number Selection
st.subheader("Automated Cluster Number Selection")
auto_determine_clusters = st.checkbox("Automatically determine the number of clusters (within 3-6)?")
if auto_determine_clusters and not st.session_state['n_clusters_determined']:
    silhouette_scores = {}
    with st.spinner("Determining optimal number of clusters..."):
        for n_clusters in range(3, 7):
            df_temp = apply_kprototypes(df.copy(), categorical_cols, n_clusters)
            # Ensure there's more than one cluster for silhouette score
            if len(df_temp['Cluster'].unique()) > 1:
                # Separate numerical and categorical data for silhouette score
                numerical_data = df_temp.drop('Cluster', axis=1).select_dtypes(include=np.number)
                if not numerical_data.empty:
                    silhouette_avg = silhouette_score(numerical_data, df_temp['Cluster'])
                    silhouette_scores[n_clusters] = silhouette_avg
                else:
                    st.warning(f"No numerical data to calculate silhouette score for {n_clusters} clusters.")
            else:
                silhouette_scores[n_clusters] = -1 # Indicate invalid

    if silhouette_scores:
        best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.success(f"Optimal number of clusters found: {best_n_clusters} (Silhouette Score: {silhouette_scores[best_n_clusters]:.4f})")
        st.session_state['n_clusters_value'] = best_n_clusters
    else:
        st.warning("Could not determine optimal number of clusters. Using default (3).")
        st.session_state['n_clusters_value'] = 3
    st.session_state['n_clusters_determined'] = True
    st.session_state['n_clusters_value'] = best_n_clusters
elif not auto_determine_clusters:
    st.session_state['n_clusters_determined'] = False
    st.session_state['n_clusters_value'] = st.slider("Select Number of Clusters for Segmentation", min_value=3, max_value=6, value=3, step=1)

if st.button("Segment and Analyze"):
    n_clusters_to_use = st.session_state['n_clusters_value']
    st.write(f"Using {n_clusters_to_use} clusters for segmentation.")
    df_segmented = apply_kprototypes(df.copy(), categorical_cols, n_clusters_to_use)
    st.session_state.segmented_df = df_segmented  # Save the segmented DataFrame
    st.session_state.clusters = sorted(df_segmented['Cluster'].unique())

    X = df_segmented.drop('Cluster', axis=1)
    y = df_segmented['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write(f"F1 Score for K-Prototype: {f1:.4f}")

    # SHAP analysis
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, model_output='raw')
    shap_values = explainer.shap_values(X_test)
    st.session_state.shap_values = shap_values
    st.session_state.X_test = X_test
    st.session_state.feature_names = list(X.columns)

    # Generate cluster descriptions using AI based on SHAP
    st.session_state.cluster_descriptions_ai = {}
    with st.spinner("Generating cluster descriptions and marketing suggestions from SHAP..."):
        for cluster_id in st.session_state.clusters:
            description = generate_cluster_description_from_shap(
                st.session_state.shap_values[:,:,1],
                st.session_state.X_test,
                st.session_state.feature_names,
                cluster_id,
                df_segmented.copy()
            )
            st.session_state.cluster_descriptions_ai[f"Cluster {cluster_id}"] = description

    st.subheader("Generated Cluster Descriptions and Marketing Suggestions:")
    for cluster, description in st.session_state.cluster_descriptions_ai.items():
        st.markdown(f"**{cluster}:**")
        st.write(description)

    # Visualization options
    show_classification_report = st.checkbox("Show Classification Report")
    show_feature_importance = st.checkbox("Show Feature Importance Bar Plot")
    show_shap_summary = st.checkbox("Show SHAP Summary Plot")
    show_shap_dependence = st.checkbox("Show SHAP Dependence Plot")

    if show_classification_report:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.table(report_df)

    if show_feature_importance:
        mean_shap = np.abs(st.session_state.shap_values[:,:,1]).mean(axis=0)
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame(mean_shap, index=st.session_state.feature_names, columns=['SHAP Value'])
        feature_importance = feature_importance.sort_values('SHAP Value', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_importance)), feature_importance['SHAP Value'])
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance.index)
        ax.set_xlabel('mean(|SHAP value|)')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
        plt.close()

    if show_shap_summary:
        st.subheader("SHAP Summary Plot")
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            st.session_state.shap_values[:,:,1],
            st.session_state.X_test,
            feature_names=st.session_state.feature_names,
            max_display=25,
            plot_type="dot",
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    if show_shap_dependence:
        mean_shap = np.abs(st.session_state.shap_values[:,:,1]).mean(axis=0)
        top_feature_idx = np.argmax(mean_shap)
        top_feature_name = st.session_state.feature_names[top_feature_idx]
        st.subheader(f"SHAP Dependence Plot - {top_feature_name}")
        fig_dep, ax_dep = plt.subplots(figsize=(12, 8))
        shap.dependence_plot(
            top_feature_idx,
            st.session_state.shap_values[:,:,1],
            st.session_state.X_test,
            feature_names=st.session_state.feature_names,
            show=False,
            ax=ax_dep
        )
        st.pyplot(fig_dep)
        plt.close()

    # AI Message Generator Section
    st.subheader("AI Message Generator")
    selected_cluster_ai = st.selectbox("Select a Cluster for Message Generation:", list(st.session_state.cluster_descriptions_ai.keys()))
    user_name_ai = st.text_input("User Name (Optional):")
    user_case_ai = st.text_area("Enter your requirement for the marketing message:")

    if st.button("Generate Marketing Message"):
        if not user_case_ai:
            st.error("Please enter your requirement for the message.")
        else:
            cluster_description_ai = st.session_state.cluster_descriptions_ai[selected_cluster_ai]
            with st.spinner("Generating marketing message..."):
                marketing_text = generate_ai_message(cluster_description_ai, user_name_ai, user_case_ai)
                st.markdown(f"### Generated Message for {selected_cluster_ai}:")
                st.write(marketing_text)
