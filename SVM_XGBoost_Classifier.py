import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Load dataset
file_path = r"D:\unitec\MachineLearningCourse\Thesis_code\combined_classified_dataset.csv"  # Use raw string (r"") for Windows paths
df = pd.read_csv(file_path, dtype=str, low_memory=False)

# ✅ Text Preprocessing Function
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

# ✅ Remove empty tweets
df = df[df['tweet'].str.strip() != ""]

# ✅ Encode category labels
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
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Train SVM Model
print("\n🔄 Training SVM Model...")
svm_model = SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train_tfidf, y_train)
print("✅ SVM Model Trained!")

# ✅ Predictions
svm_predictions = svm_model.predict(X_test_tfidf)

# ✅ Evaluate Model Performance
print("\n📌 SVM Model Performance:")
print(classification_report(y_test, svm_predictions))
accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {accuracy:.4f}")

# ✅ Confusion Matrix
conf_matrix = confusion_matrix(y_test, svm_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=category_mapping.keys(), yticklabels=category_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

# ✅ Compute Correlation Matrix
correlation_matrix = pd.DataFrame(X_train_tfidf.toarray()).corr()

# ✅ Plot Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Matrix")
plt.show()

print("\n✅ Model Training & Evaluation Completed!")
