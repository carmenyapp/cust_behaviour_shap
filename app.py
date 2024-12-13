import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('cluster_marketing_campaign.csv')

# Define the target columns for each option
option_1 = ['MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

option_2 = ['NumDealsPurchases', 'NumWebPurchases', 
            'NumCatalogPurchases', 'NumStorePurchases']

option_3 = ['NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 
            'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 
            'Complain', 'Response']

# User selection for target columns
target_option = st.radio("Select Target Columns to Analyze", 
                         ("Spending Categories", "Purchase Metrics", "Response Metrics"))

# Assign target columns based on user selection
if target_option == "Spending Categories":
    target_columns = option_1
elif target_option == "Purchase Metrics":
    target_columns = option_2
else:
    target_columns = option_3

encoder = LabelEncoder()

# Binning for option_1 and option_2, but not for option_3
for col in target_columns:
    if col not in option_3:  # Skip binning for option_3 columns
        df[f'{col}_binned'] = pd.qcut(df[col], q=3, labels=['low', 'medium', 'high'])
        df[f'{col}_encoded'] = encoder.fit_transform(df[[f'{col}_binned']])
    else:
        df[f'{col}_encoded'] = encoder.fit_transform(df[[col]])  # Encode binary columns for option_3

# Feature preparation
features = df.drop(columns=[f'{col}_binned' for col in target_columns if col not in option_3])  # Drop binned columns for option_3
encoded_target = [f'{col}_encoded' for col in target_columns]

shap_results = {}

# Analysis loop
for target in encoded_target:
    st.write(f"Training for Target: {target}")

    # Train-Test Split for each target
    X = features
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    st.write(f"F1 Score for {target}: {f1}")
    st.text(report)

    # SHAP values and plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_results[target] = {
        'shap_values': shap_values,
        'f1_score': f1,
        'report': report
    }

    # SHAP summary plot
    st.write(f"SHAP Summary Plot for {target}")
    plot_width = st.slider("Select Plot Width", min_value=5, max_value=20, value=12)
    plot_height = st.slider("Select Plot Height", min_value=5, max_value=15, value=8)
    plt.figure(figsize=(plot_width, plot_height)) 
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches="tight")
