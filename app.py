import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('cluster_marketing_campaign.csv')

option_1 = ['MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

option_2 = ['NumDealsPurchases', 'NumWebPurchases', 
            'NumCatalogPurchases', 'NumStorePurchases']

option_3 = ['NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 
            'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 
            'Complain', 'Response']

target_option = st.radio("Select Target Columns to Analyze", 
                         ("Spending Categories", "Purchase Metrics", "Response Metrics"))

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

features = df.drop(columns=[f'{col}_binned' for col in target_columns if col not in option_3])  # Drop binned columns for option_3
encoded_target = [f'{col}_encoded' for col in target_columns]

shap_results = {}
results_table = []
for target in encoded_target:
    st.write(f"Training for Target: {target}")

    # Train-Test Split for each target
    X = features  
    y = df[target]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    # Predict and calculate F1 score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    results_table.append({
        "Target": target,
        "F1 Score": f1,
        "Precision (Class 0)": report['0']['precision'],
        "Recall (Class 0)": report['0']['recall'],
        "F1 Score (Class 0)": report['0']['f1-score'],
        "Precision (Class 1)": report['1']['precision'],
        "Recall (Class 1)": report['1']['recall'],
        "F1 Score (Class 1)": report['1']['f1-score'],
        "Macro avg Precision": report['macro avg']['precision'],
        "Macro avg Recall": report['macro avg']['recall'],
        "Macro avg F1 Score": report['macro avg']['f1-score'],
        "Weighted avg Precision": report['weighted avg']['precision'],
        "Weighted avg Recall": report['weighted avg']['recall'],
        "Weighted avg F1 Score": report['weighted avg']['f1-score']
    })

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap_results[target] = {
        'shap_values': shap_values,
        'f1_score': f1,
        'report': report
    }

    st.write(f"SHAP Summary Plot for {target}")
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0].values, X_test, show=False)
    else:
        shap.summary_plot(shap_values.values, X_test, show=False)

    st.pyplot(bbox_inches="tight")

results_df = pd.DataFrame(results_table)
st.write("Model Performance Results:", results_df)
