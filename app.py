import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, silhouette_score
import shap
import json
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

def initialize_session_state():
    """
    Initialize or reset all necessary session state variables.
    """
    default_states = {
        'n_clusters_determined': False,
        'n_clusters_value': 3,
        'segmented_df': None,
        'clusters': [],
        'shap_values': None,
        'X_test': None,
        'feature_names': [],
        'cluster_descriptions_ai': {},
        'model': None,
        'X_train': None,
        'y_train': None,
        'y_test': None
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def determine_optimal_clusters(df, categorical_cols):
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
                silhouette_scores[n_clusters] = -1  # Indicate invalid

    if silhouette_scores:
        best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.success(f"Optimal number of clusters found: {best_n_clusters} (Silhouette Score: {silhouette_scores[best_n_clusters]:.4f})")
        return best_n_clusters
    else:
        st.warning("Could not determine optimal number of clusters. Using default (3).")
        return 3

def apply_kprototypes(df, categorical_cols, n_clusters):
    # K-Prototypes clustering
    kproto = KPrototypes(n_clusters=n_clusters, init='Huang', random_state=42)
    categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]
    clusters = kproto.fit_predict(df, categorical=categorical_indices)
    df['Cluster'] = clusters
    return df
    
def perform_clustering_analysis(df, categorical_cols, n_clusters_to_use):
    # Perform clustering
    df_segmented = apply_kprototypes(df.copy(), categorical_cols, n_clusters_to_use)
    
    # Prepare data for model training
    X = df_segmented.drop('Cluster', axis=1)
    y = df_segmented['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write(f"F1 Score for K-Prototype: {f1:.4f}")

    # SHAP Analysis
    for cluster in df_segmented['Cluster'].unique():
        df_segmented['binary_target'] = (df_segmented['Cluster'] == cluster).astype(int)
            
        # Features and target
        X = df_segmented.drop(columns=['Cluster', 'binary_target'])
        y = df_segmented['binary_target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        explainer = shap.TreeExplainer(model, model_output='raw')
        shap_values = explainer.shap_values(X_test)
    
    return shap_values, X_test, list(X.columns), y_test
    
def generate_cluster_descriptions(shap_values, X_test, feature_names, n_clusters):
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    if isinstance(X_test, list):
        X_test = np.array(X_test)
    
    # Comprehensive cluster description dictionary
    cluster_descriptions = {}
    
    if len(shap_values.shape) == 2:
        # If 2D, assume single cluster scenario
        shap_values = shap_values[:, :, np.newaxis]
    
    # Ensure n_clusters doesn't exceed available clusters
    max_clusters = shap_values.shape[2]
    n_clusters = min(n_clusters, max_clusters)
    
    for cluster in range(n_clusters):
        try:
            # 1. Feature Importance Ranking
            binary_shap = shap_values[:,:,cluster]
            mean_abs_shap = np.abs(binary_shap).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            # 2. Feature Direction Analysis
            mean_shap = binary_shap.mean(axis=0)
            feature_direction = pd.DataFrame({
                'feature': feature_names,
                'mean_shap': mean_shap,
                'direction': np.where(mean_shap > 0, 'Positive', 'Negative')
            })
            
            # 3. Feature Distribution Within Cluster
            feature_distribution = {}
            for feature in feature_names:
                feature_idx = feature_names.index(feature)
                feature_shap_values = binary_shap[:, feature_idx]
                feature_distribution[feature] = {
                    'mean': float(np.mean(feature_shap_values)),
                    'median': float(np.median(feature_shap_values)),
                    'std': float(np.std(feature_shap_values)),
                    'range': (float(np.min(feature_shap_values)), float(np.max(feature_shap_values))),
                    'percentiles': {
                        '25th': float(np.percentile(feature_shap_values, 25)),
                        '75th': float(np.percentile(feature_shap_values, 75))
                    }
                }
            
            # Prepare comprehensive analysis for AI prompt
            analysis_prompt = f"""
            Cluster {cluster} Detailed Analysis:

            Feature Importance Ranking:
            {feature_importance.to_string()}

            Feature Direction Analysis:
            {feature_direction.to_string()}

            Feature Distribution Details:
            {json.dumps(feature_distribution, indent=2)}

            Task: Generate a comprehensive, insightful description of this customer cluster based on the SHAP analysis. 
            Provide a narrative that transforms these technical metrics into a meaningful, actionable cluster profile.

            Focus on:
            1. Key distinguishing features of this cluster
            2. Impact and direction of top features
            3. Unique characteristics that define this cluster
            4. Potential business insights or actionable strategies

            Your description should be:
            - Concise (250-350 words)
            - Data-driven
            - Actionable
            - Easy to understand for business stakeholders
            """
            
            # Generate AI Description
            cluster_description = generate_ai_cluster_description(analysis_prompt)
            
            cluster_descriptions[f'Cluster {cluster}'] = cluster_description
        
        except Exception as e:
            cluster_descriptions[f'Cluster {cluster}'] = f"Error generating description for cluster {cluster}: {str(e)}"
    
    return cluster_descriptions

def generate_ai_description(cluster_descriptions):
    try:
        # Use OpenAI API to generate description
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst describing a cluster's characteristics."},
                {"role": "user", "content": cluster_descriptions}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating description: {str(e)}"

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

def ai_message_generator(cluster_descriptions):
    st.subheader("AI Message Generator")
    selected_cluster_ai = st.selectbox("Select a Cluster for Message Generation:", list(cluster_descriptions.keys()))
    user_name_ai = st.text_input("User Name (Optional):")
    user_case_ai = st.text_area("Enter your requirement for the marketing message:")

    if st.button("Generate Marketing Message"):
        if not user_case_ai:
            st.error("Please enter your requirement for the message.")
        else:
            cluster_description_ai = cluster_descriptions[selected_cluster_ai]
            with st.spinner("Generating marketing message..."):
                marketing_text = generate_ai_message(cluster_description_ai, user_name_ai, user_case_ai)
                st.markdown(f"### Generated Message for {selected_cluster_ai}:")
                st.write(marketing_text)


#  my code start here

initialize_session_state()
df, categorical_cols = load_data()
st.title("AI Message Generation from Customer Analysis")

st.subheader("Cluster Number Selection")
auto_determine_clusters = st.checkbox("Automatically determine the number of clusters (within 3-6)?")

if auto_determine_clusters and not st.session_state['n_clusters_determined']:
    best_n_clusters = determine_optimal_clusters(df, categorical_cols)
    st.session_state['n_clusters_value'] = best_n_clusters
    st.session_state['n_clusters_determined'] = True
elif not auto_determine_clusters:
    st.session_state['n_clusters_determined'] = False
    st.session_state['n_clusters_value'] = st.slider("Select Number of Clusters for Segmentation", min_value=3, max_value=6, value=3, step=1)
     
if st.button("Segment and Analyze"):
    # Perform clustering & shap analysis
    (
        st.session_state['shap_values'], 
        st.session_state['X_test'], 
        st.session_state['feature_names'],
        st.session_state['y_test']
    ) = perform_clustering_analysis(df, categorical_cols, st.session_state['n_clusters_value'])

    # Generate cluster descriptions
    st.session_state['cluster_descriptions_ai'] = generate_cluster_descriptions(
        st.session_state['shap_values'], 
        st.session_state['X_test'], 
        st.session_state['feature_names'],
        st.session_state['n_clusters_value']
    )

# Display Results if Available
if st.session_state.get('cluster_descriptions_ai'):
    # Display Cluster Descriptions
    for cluster, description in st.session_state.cluster_descriptions_ai.items():
        st.subheader(f"{cluster} Description")
        st.write(description)

    # Create Visualization Options
    show_classification_report = st.checkbox("Show Classification Report")
    show_feature_importance = st.checkbox("Show Feature Importance Bar Plot")
    show_shap_summary = st.checkbox("Show SHAP Summary Plot")
    show_shap_dependence = st.checkbox("Show SHAP Dependence Plot")

    # Display Selected Visualizations
    if show_classification_report:
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
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

    # AI Message Generator
    ai_message_generator(st.session_state['cluster_descriptions_ai'])
