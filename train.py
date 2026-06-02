import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("data.csv")

df["description"] = df["description"].fillna("")
df["description"] = df["description"].str.lower()

X = df["description"]
y_category = df["category"]
y_priority = df["priority"]

# 🔥 Better feature extraction (IMPORTANT)
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english",
    max_df=0.85,
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(X)

# 🔥 Stronger models
model_category = LogisticRegression(max_iter=2000, class_weight="balanced")
model_priority = LogisticRegression(max_iter=2000, class_weight="balanced")

model_category.fit(X_vec, y_category)
model_priority.fit(X_vec, y_priority)

# Save models
joblib.dump(model_category, "model_category.pkl")
joblib.dump(model_priority, "model_priority.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("🚀 FULL MODEL TRAINED SUCCESSFULLY")