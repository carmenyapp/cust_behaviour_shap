import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

# Define target columns for option 1 and option 2
option_1 = ['MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

option_2 = ['NumDealsPurchases', 'NumWebPurchases', 
            'NumCatalogPurchases', 'NumStorePurchases']
option_3 = ['NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 
            'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 
            'Complain', 'Response']

target_option = st.radio("Select Target Columns to Analyze", ("Spending Categories", "Purchase Metrics","Response Metrics"))

if target_option == "Spending Categories":
    target_columns = option_1
else if target_option == "Purchase Metrics":
    target_columns = option_2
else:
    target_columns = option_3

encoder = LabelEncoder()
for col in target_columns:
    df[f'{col}_binned'] = pd.qcut(df[col], q=3, labels=['low', 'medium', 'high'])
    df[f'{col}_encoded'] = encoder.fit_transform(df[[f'{col}_binned']])

# Feature preparation
features = df.drop(columns=[f'{col}_binned' for col in target_columns])
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
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches="tight")
