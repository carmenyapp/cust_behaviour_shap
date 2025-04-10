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
    aX = df_segmented.drop('Cluster', axis=1)
    ay = df_segmented['Cluster']
    aX_train, aX_test, ay_train, ay_test = train_test_split(aX, ay, test_size=0.2, random_state=42)
    
    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(aX_train, ay_train)
    ay_pred = clf.predict(aX_test)
    
    # Calculate F1 Score
    f1 = f1_score(ay_test, ay_pred, average='weighted')
    st.write(f"F1 Score for K-Prototype: {f1:.4f}")

    # SHAP Analysis
    shap_results = {}
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

        shap_results[cluster] = {
            'shap_values': shap_values,
            'X_test': X_test,
            'feature_names': list(X.columns),
            'y_test': y_test
        }

    return shap_results, ay_test, ay_pred

def cluster_descriptions_generator(shap_values, X_test, feature_names, cluster_id):
    try:
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]
        
        # 1. Feature Importance Ranking
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # 2. Feature Direction Analysis
        mean_shap = shap_values.mean(axis=0)
        feature_direction = pd.DataFrame({
            'feature': feature_names,
            'mean_shap': mean_shap,
            'direction': np.where(mean_shap > 0, 'Positive', 'Negative')
        })
        
        # 3. Feature Distribution Within Cluster
        feature_distribution = {}
        for feature in feature_names:
            feature_idx = feature_names.index(feature)
            feature_shap_values = shap_values[:, feature_idx]
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
        Cluster {cluster_id} SHAP-Based Customer Segment Analysis
        
        --- FEATURE INSIGHTS ---
        
        Top Feature Importance (mean absolute SHAP values):
        {feature_importance.to_string(index=False)}
        
        Top Feature Directions (mean SHAP values):
        {feature_direction.to_string(index=False)}
        
        Feature Distribution Summary (SHAP values):
        {json.dumps(feature_distribution, indent=2)}
        
        --- TASK INSTRUCTIONS ---
        
        You are a business analyst AI assistant. Based on the SHAP analysis above, complete the two tasks below.
        
        ### TASK 1: CLUSTER NAME
        Generate a short, descriptive cluster name (3–6 words) that captures the essence of this customer segment.
        
        **Requirements:**
        - Reflect key impactful features and their directions
        - Be catchy, business-relevant, and unique
        - Avoid generic terms unless truly applicable
        - Format the name on its own line, exactly like this:
        `CLUSTER_NAME: Your Descriptive Name`
        
        ### TASK 2: CLUSTER DESCRIPTION
        Generate a concise, professional description of the cluster. 
        
        **Requirements:**
        - Length: < 350 words
        - Style: Business-friendly and insightful
        - Must explain:
          - What defines this cluster (key traits)
          - How top features shape behavior (direction & impact)
          - Business insights or strategies for engagement
        - Format: Start right after the name on a new paragraph.
        
        ### FORMAT REQUIREMENTS:
        Return your response **exactly** as follows:
        Do not include any extra explanation or commentary.
        """
        
        # Generate AI Description
        full_response = generate_ai_description(analysis_prompt)
        
        # Parse the response to separate name and description
        # Assuming format: "CLUSTER_NAME: Name\n\nDescription..."
        try:
            if isinstance(full_response, str) and "CLUSTER_NAME:" in full_response:
                parts = full_response.split('\n\n', 1)
                cluster_name = parts[0].replace("CLUSTER_NAME:", "").strip()
                description = parts[1].strip() if len(parts) > 1 else "No description generated."
            else:
                # If CLUSTER_NAME is not present, use first line as name and rest as description
                lines = full_response.strip().split('\n', 1)
                cluster_name = lines[0].strip()
                description = lines[1].strip() if len(lines) > 1 else "No description provided."
        except Exception:
            # Fallback if anything fails in parsing
            description = full_response
            try:
                top_feature = feature_importance.iloc[0]['feature']
                top_direction = feature_direction[feature_direction['feature'] == top_feature]['direction'].values[0]
                cluster_name = f"{top_direction} {top_feature} Cluster"
            except Exception:
                cluster_name = "Unnamed Cluster"
        return {
            'name': cluster_name,
            'description': description
        }
        
    except Exception as e:
        return f"Error generating description for cluster {cluster_id}: {str(e)}"

def generate_clusters_description(shap_results):
    cluster_info = {}
    for cluster_id, result in shap_results.items():
        info = cluster_descriptions_generator(
            result['shap_values'],
            result['X_test'],
            result['feature_names'],
            cluster_id
        )
        cluster_info[cluster_id] = info
    
    return cluster_info

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
    selected_cluster_ai = st.selectbox("Select a Cluster for Message Generation:", list(sorted(cluster_descriptions.keys())))
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
manual_n_clusterd = st.checkbox("Manually determine the number of clusters (within 3-6)?")

if not manual_n_clusterd and st.session_state['n_clusters_determined']:
    best_n_clusters = determine_optimal_clusters(df, categorical_cols)
    st.session_state['n_clusters_value'] = best_n_clusters
    st.session_state['n_clusters_determined'] = True
elif manual_n_clusterd:
    st.session_state['n_clusters_determined'] = False
    st.session_state['n_clusters_value'] = st.slider("Select Number of Clusters for Segmentation", min_value=3, max_value=6, value=3, step=1)
     
if st.button("Segment and Analyze"):
    # Perform clustering & shap analysis
    (
        st.session_state['shap_results'], 
        st.session_state.y_test, 
        st.session_state.y_pred
    )= perform_clustering_analysis(df, categorical_cols, st.session_state['n_clusters_value'])

    # Generate cluster descriptions
    cluster_info = generate_clusters_description(st.session_state['shap_results'])
    st.session_state['cluster_descriptions_ai'] = cluster_info
    
# Display Results if Available
if st.session_state.get('cluster_descriptions_ai'):
    sorted_cluster_ids = sorted(st.session_state['cluster_descriptions_ai'].keys())
    # Display Cluster Descriptions
    for cluster_id in sorted_cluster_ids:
        info = st.session_state['cluster_descriptions_ai'][cluster_id]
        if isinstance(info, dict) and 'name' in info and 'description' in info:
            st.subheader(f"Cluster {cluster_id}: {info['name']}")
            st.write(info['description'])
        elif isinstance(info, str):
            st.subheader(f"Cluster {cluster_id}")
            st.write(info)
        # Fallback for any other unexpected structure
        else:
            st.subheader(f"Cluster {cluster_id}")
            st.write("Error: Unable to display cluster information in expected format.")
            st.write(info)
        
    st.sidebar.subheader("Visualization Options")
    show_classification_report = st.sidebar.checkbox("Show Classification Report")
    show_shap_summary = st.sidebar.checkbox("Show SHAP Summary Plot")
    
    # Display Selected Visualizations
    if show_classification_report:
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("Clustering Result Classification Report:")
        st.table(report_df)

    if show_shap_summary:
        st.subheader("SHAP Summary Plots by Cluster")

        for cluster_id, result in st.session_state['shap_results'].items():
            st.markdown(f"### Cluster {cluster_id}")
        
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(
                result['shap_values'][:, :, 1],  
                result['X_test'],
                feature_names=result['feature_names'],
                max_display=25,
                plot_type="dot",
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

    # AI Message Generator
    ai_message_generator(st.session_state['cluster_descriptions_ai'])
