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

@st.cache_data
def perform_clustering_and_analysis(df, categorical_cols, n_clusters_to_use):
    """
    Perform clustering, model training, and SHAP analysis with caching
    """
    # Apply K-Prototypes clustering
    df_segmented = apply_kprototypes(df.copy(), categorical_cols, n_clusters_to_use)
    
    # Prepare data for model training
    X = df_segmented.drop('Cluster', axis=1)
    y = df_segmented['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # SHAP analysis
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, model_output='raw')
    shap_values = explainer.shap_values(X_test)

    return {
        'df_segmented': df_segmented,
        'clusters': sorted(df_segmented['Cluster'].unique()),
        'X_test': X_test,
        'shap_values': shap_values,
        'feature_names': list(X.columns),
        'f1_score': f1,
        'y_test': y_test,
        'y_pred': y_pred
    }

@st.cache_data
def generate_cluster_descriptions(shap_values, X_test, feature_names, num_clusters):
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    X_test = np.array(X_test)
    
    # Calculate the absolute mean SHAP values for each feature
    if shap_values.ndim == 3:
        # Multiclass or multi-output case
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
    else:
        # Binary or single-output case
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    top_feature_indices = mean_abs_shap.argsort()[::-1]

    cluster_descriptions_ai = {}
    
    for cluster in range(num_clusters):
        # Select SHAP values for this cluster
        if shap_values.ndim == 3:
            cluster_shap = shap_values[0, cluster, :]  # Adjust indexing as needed
        else:
            cluster_shap = shap_values[cluster]
        
        # Prepare description components
        top_positive_features = []
        top_negative_features = []
        
        for idx in top_feature_indices[:5]:  # Top 5 features
            feature_name = feature_names[idx]
            try:
                feature_shap_value = float(cluster_shap[idx])
            except Exception as e:
                print(f"Error processing feature {feature_name}: {e}")
                continue
            
            if feature_shap_value > 0:
                top_positive_features.append({
                    'name': feature_name,
                    'impact': feature_shap_value
                })
            else:
                top_negative_features.append({
                    'name': feature_name,
                    'impact': abs(feature_shap_value)
                })
        
        # Sort features by absolute impact
        top_positive_features.sort(key=lambda x: x['impact'], reverse=True)
        top_negative_features.sort(key=lambda x: x['impact'], reverse=True)
        
        # Generate AI description
        description = generate_ai_description(
            top_positive_features, 
            top_negative_features
        )
        
        cluster_descriptions_ai[f"Cluster {cluster}"] = description
    
    return cluster_descriptions_ai

def generate_ai_description(positive_features, negative_features):
    # Prepare the prompt
    prompt = f"""Generate a concise and insightful description of a data cluster based on its most important features.

    Positive Influential Features:
    {', '.join([f"{f['name']} (impact: {f['impact']:.2f})" for f in positive_features])}
    
    Negative Influential Features:
    {', '.join([f"{f['name']} (impact: {f['impact']:.2f})" for f in negative_features])}
    
    Description:"""

    try:
        # Use OpenAI API to generate description
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst describing a cluster's characteristics."},
                {"role": "user", "content": prompt}
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

def clustering_analysis():
    # Streamlit UI Configuration
    st.title("Message Generation from Customer Analysis")
    
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
        with st.spinner("Performing clustering and analysis..."):
        n_clusters_to_use = st.session_state['n_clusters_value']
        analysis_results = perform_clustering_and_analysis(
                    df,
                    categorical_cols, 
                    n_clusters_to_use
                )
                # Store results in session state for later use
                st.session_state.analysis_results = analysis_results
                st.session_state.clusters = analysis_results['clusters']
                
                # Redirect or show success message
                st.success("Analysis complete! Proceed to detailed analysis.")

def result_text_generator():
    st.title("Detail Analysis & AI Text Generator")

    if 'analysis_results' not in st.session_state:
        st.warning("Please perform customer analysis first.")
        return

    # Retrieve cached analysis results
    analysis_results = st.session_state.analysis_results

    # Display basic clustering information
    st.write(f"Using {len(analysis_results['clusters'])} clusters")
    st.write(f"F1 Score: {analysis_results['f1_score']:.4f}")

    # Cluster Descriptions
    st.header("Cluster Descriptions")
    if 'cluster_descriptions_ai' not in st.session_state:
        # Generate descriptions if not already done
        with st.spinner("Generating cluster descriptions..."):
            st.session_state.cluster_descriptions_ai = generate_cluster_descriptions_cached(
                analysis_results['shap_values'], 
                analysis_results['X_test'], 
                analysis_results['feature_names'],
                len(analysis_results['clusters'])
            )

    # Display cluster descriptions
    for cluster, description in st.session_state.cluster_descriptions_ai.items():
        st.subheader(f"Cluster {cluster} Description")
        st.write(description)

    # Visualization and Analysis Options
    st.header("Analysis Visualizations")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_classification_report = st.checkbox("Classification Report")
    with col2:
        show_feature_importance = st.checkbox("Feature Importance")
    with col3:
        show_shap_summary = st.checkbox("SHAP Summary")
    with col4:
        show_shap_dependence = st.checkbox("SHAP Dependence")

    # Visualization Logic (similar to previous implementation)
    if show_classification_report:
        report = classification_report(
            analysis_results['y_test'], 
            analysis_results['y_pred'], 
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.table(report_df)

    if show_feature_importance:
        mean_shap = np.abs(analysis_results['shap_values'][:,:,1]).mean(axis=0)
        feature_importance = pd.DataFrame(
            mean_shap, 
            index=analysis_results['feature_names'], 
            columns=['SHAP Value']
        ).sort_values('SHAP Value', ascending=True)

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
            analysis_results['shap_values'][:,:,1],
            analysis_results['X_test'],
            feature_names=analysis_results['feature_names'],
            max_display=25,
            plot_type="dot",
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    if show_shap_dependence:
        mean_shap = np.abs(analysis_results['shap_values'][:,:,1]).mean(axis=0)
        top_feature_idx = np.argmax(mean_shap)
        top_feature_name = analysis_results['feature_names'][top_feature_idx]
        
        st.subheader(f"SHAP Dependence Plot - {top_feature_name}")
        fig_dep, ax_dep = plt.subplots(figsize=(12, 8))
        shap.dependence_plot(
            top_feature_idx,
            analysis_results['shap_values'][:,:,1],
            analysis_results['X_test'],
            feature_names=analysis_results['feature_names'],
            show=False,
            ax=ax_dep
        )
        st.pyplot(fig_dep)
        plt.close()

    # AI Message Generator Section
    st.header("AI Message Generator")
    selected_cluster_ai = st.selectbox(
        "Select a Cluster for Message Generation:", 
        list(st.session_state.cluster_descriptions_ai.keys())
    )
    user_name_ai = st.text_input("User Name (Optional):")
    user_case_ai = st.text_area("Enter your requirement for the marketing message:")

    if st.button("Generate Marketing Message"):
        if not user_case_ai:
            st.error("Please enter your requirement for the message.")
        else:
            cluster_description_ai = st.session_state.cluster_descriptions_ai[selected_cluster_ai]
            with st.spinner("Generating marketing message..."):
                marketing_text = generate_ai_message(
                    cluster_description_ai, 
                    user_name_ai, 
                    user_case_ai
                )
                st.markdown(f"### Generated Message for Cluster {selected_cluster_ai}:")
                st.write(marketing_text)

if 'analysis_results' not in st.session_state:
    clustering_analysis()
else:
    result_text_generator()
