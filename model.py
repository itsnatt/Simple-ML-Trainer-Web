import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
def train_and_evaluate(df, x_column, y_column, max_features, ngram_range, model_name, model_params):
    X = df[x_column].astype(str)
    y = df[y_column]

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Pilih model berdasarkan input user
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=model_params.get("n_neighbors", 5))
    elif model_name == "SVM":
        model = SVC(C=model_params.get("C", 1.0), kernel="linear", probability=True)
    elif model_name == "Naive Bayes":
        model = MultinomialNB()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=2000, C=model_params.get("C", 1.0))
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=model_params.get("n_estimators", 100))

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ✅ Perbaiki: Tambahkan y_pred
    report = classification_report(y_test, y_pred, output_dict=False)  
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Simpan model
    model_filename = os.path.join(MODEL_DIR, f"{model_name.lower()}.pkl")
    joblib.dump(model, model_filename)

    # Simpan confusion matrix sebagai gambar
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig("static/conf_matrix.png")
    plt.close()

    return report, accuracy, conf_matrix, model_filename
