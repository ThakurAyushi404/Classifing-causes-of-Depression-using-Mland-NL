import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Load the balanced dataset
file_path = "balanced_dataset.csv"
df = pd.read_csv(file_path, dtype=str, low_memory=False)

# ✅ Text preprocessing function (ensures proper data formatting)
def clean_text(text):
    if pd.isnull(text) or not isinstance(text, str):
        return ""  # Convert invalid values to empty strings
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions (@username)
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters except spaces
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces and trim

    return text if len(text.split()) > 2 else ""  # Ensure tweets contain more than 2 words

# ✅ Apply text cleaning
df['tweet'] = df['tweet'].astype(str).apply(clean_text)

# ✅ Remove empty rows (tweets that became blank after cleaning)
df = df[df['tweet'].str.strip() != ""]

# ✅ Convert category labels into numbers
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print(f"🔹 Category Mapping: {category_mapping}")

# ✅ Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])
print(f"✅ Train size: {len(X_train)}, Test size: {len(X_test)}")

# ✅ Convert text into numerical representation using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# ✅ Ensure vocabulary exists before proceeding
if len(vectorizer.get_feature_names_out()) == 0:
    raise ValueError("TF-IDF Vocabulary is empty! Check text preprocessing and dataset.")

X_test_tfidf = vectorizer.transform(X_test)

# ✅ Compute Class Weights for Imbalanced Data
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}

print(f"\n🔹 Class Weights for Training: {class_weights}")

# ✅ Train SVM with Class Weights
print("\n🔄 Training SVM with Class Weights...")
svm_model = SVC(kernel='linear', C=1.0, probability=True, class_weight=class_weights)
svm_model.fit(X_train_tfidf, y_train)
print("✅ SVM Model Trained!")

# ✅ Train XGBoost with Class Weights
print("\n🔄 Training XGBoost with Class Weights...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=list(class_weights.values()),
                          max_depth=5, learning_rate=0.05, n_estimators=200)
xgb_model.fit(X_train_tfidf, y_train)
print("✅ XGBoost Model Trained!")

# ✅ Predictions
svm_predictions = svm_model.predict(X_test_tfidf)
xgb_predictions = xgb_model.predict(X_test_tfidf)

# ✅ Evaluate SVM
print("\n📌 SVM Model Performance:")
print(classification_report(y_test, svm_predictions))
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))

# ✅ Evaluate XGBoost
print("\n📌 XGBoost Model Performance:")
print(classification_report(y_test, xgb_predictions))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_predictions))

# ✅ Confusion Matrix Plot for SVM
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, svm_predictions), annot=True, fmt='d', cmap='Blues',
            xticklabels=category_mapping.keys(), yticklabels=category_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

# ✅ Confusion Matrix Plot for XGBoost
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, xgb_predictions), annot=True, fmt='d', cmap='Oranges',
            xticklabels=category_mapping.keys(), yticklabels=category_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")
plt.show()

print("\n✅ Model Training & Evaluation Completed!")
