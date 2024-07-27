import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("./datasets/Crop_recommendation.csv")

# Encode labels
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# Split data
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gaussian Naive Bayes': GaussianNB()
}

# Train models and save metrics
model_metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    model_metrics[name] = {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    # Save the model
    joblib.dump(model, f"{name}.pkl")


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load models and metrics
model_files = {
    'Logistic Regression': 'Logistic Regression.pkl',
    'SVM': 'SVM.pkl',
    'Random Forest': 'Random Forest.pkl',
    'Gradient Boosting': 'Gradient Boosting.pkl',
    'KNN': 'KNN.pkl',
    'Decision Tree': 'Decision Tree.pkl',
    'Gaussian Naive Bayes': 'Gaussian Naive Bayes.pkl'
}

model_metrics = {}
for name, path in model_files.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    model_metrics[name] = {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

# Streamlit app
st.title("Model Training Dashboard")

st.sidebar.title("Select a Model")
model_choice = st.sidebar.selectbox("Choose the model", list(model_metrics.keys()))

st.header(f"Metrics for {model_choice}")

# Display accuracy
accuracy = model_metrics[model_choice]['accuracy']
st.subheader("Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Display confusion matrix
conf_matrix = model_metrics[model_choice]['confusion_matrix']
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Display classification report
class_report = model_metrics[model_choice]['classification_report']
st.subheader("Classification Report")
st.write(pd.DataFrame(class_report).transpose())

# Additional plots like feature importance (if applicable)
if model_choice in ['Random Forest', 'Gradient Boosting']:
    st.subheader("Feature Importance")
    feature_importance = model_metrics[model_choice]['model'].feature_importances_
    sorted_idx = np.argsort(feature_importance)
    
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(np.array(X.columns)[sorted_idx])
    st.pyplot(fig)
