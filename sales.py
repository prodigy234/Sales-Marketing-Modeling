
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, silhouette_score
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Supermarket Sales Analytics", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("supermarket_sales.csv")

df = load_data()

# Sidebar
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to", ["Dataset Overview", "Exploratory Analysis", "Customer Segmentation", "Churn Prediction", "Feature Insights"])

# Dataset Overview
if section == "Dataset Overview":
    st.title("📄 Dataset Overview")
    st.write("Preview of the dataset:")
    st.dataframe(df.head())
    st.subheader("Summary")
    st.text(str(df.info()))
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

# Exploratory Analysis
elif section == "Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Total Purchase by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Total', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Sales by Product Line")
    fig, ax = plt.subplots()
    df.groupby('Product line')['Total'].sum().sort_values().plot(kind='barh', ax=ax)
    st.pyplot(fig)

# Customer Segmentation
elif section == "Customer Segmentation":
    st.title("🧮 Customer Segmentation (K-Means Clustering)")

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    df.drop(['Invoice ID', 'Date', 'Time'], axis=1, inplace=True)
    scaler = StandardScaler()
    numerical_cols = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    kmeans_data = df[['Total', 'Quantity', 'gross income']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(kmeans_data)

    sil_score = silhouette_score(kmeans_data, df['Cluster'])
    st.write(f"Silhouette Score: {sil_score:.3f}")

    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Total', y='Quantity', hue='Cluster', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

# Churn Prediction
elif section == "Churn Prediction":
    st.title("📉 Churn Prediction (Random Forest Classifier)")

    # Prepare X and y
    if 'Customer type' not in df.columns:
        st.warning("Customer type column missing.")
    else:
        X = df.drop('Customer type', axis=1)
        y = df['Customer type']
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Drop non-numeric features
        X = X.select_dtypes(include=[np.number])

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.subheader("ROC AUC Score")
            st.write(f"{roc_auc_score(y_test, y_proba):.3f}")
        except Exception as e:
            st.error(f"Error in model training: {e}")

# Feature Importance
elif section == "Feature Insights":
    st.title("📌 Feature Importance")
    try:
        X = df.drop('Customer type', axis=1)
        X = X.select_dtypes(include=[np.number])
        y = df['Customer type']
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        model = RandomForestClassifier(random_state=42).fit(X, y)

        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Top Features")
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Feature insight computation failed: {e}")
