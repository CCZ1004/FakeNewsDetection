import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

# Load the BBC News Dataset for Categorization
bbc_file_path = "C:\\Users\\USER\\OneDrive\\Documents\\NLP_Project\\bbc-news-data.csv"
bbc_df = pd.read_csv(bbc_file_path, sep='\t')

# Load Fake News Dataset
real_news_path = "C:\\Users\\USER\\OneDrive\\Documents\\NLP_Project\\True.csv"
fake_news_path = "C:\\Users\\USER\\OneDrive\\Documents\\NLP_Project\\Fake.csv"
real_news_df = pd.read_csv(real_news_path)
fake_news_df = pd.read_csv(fake_news_path)

# Preprocess Text Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Preprocessing for BBC News Dataset
bbc_df['processed_content'] = bbc_df['content'].apply(preprocess_text)

# Preprocessing for Fake News Dataset
real_news_df['label'] = 'REAL'
fake_news_df['label'] = 'FAKE'
real_news_df['processed_content'] = real_news_df['text'].apply(preprocess_text)
fake_news_df['processed_content'] = fake_news_df['text'].apply(preprocess_text)

# Combine Real and Fake News for Fake News Detection
fake_news_df_combined = pd.concat([real_news_df[['processed_content', 'label']], 
                                   fake_news_df[['processed_content', 'label']]], ignore_index=True)

# Shuffle the Fake News Dataset
fake_news_df_combined = fake_news_df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# 1. Train a News Categorization Model (BBC Dataset)
X_bbc = bbc_df['processed_content']
y_bbc = bbc_df['category']

X_train_bbc, X_test_bbc, y_train_bbc, y_test_bbc = train_test_split(X_bbc, y_bbc, test_size=0.2, random_state=42)

vectorizer_bbc = TfidfVectorizer(max_features=5000)
X_train_tfidf_bbc = vectorizer_bbc.fit_transform(X_train_bbc)
X_test_tfidf_bbc = vectorizer_bbc.transform(X_test_bbc)

model_bbc = LogisticRegression(max_iter=1000)
model_bbc.fit(X_train_tfidf_bbc, y_train_bbc)

# Evaluate the BBC Categorization Model
y_pred_bbc = model_bbc.predict(X_test_tfidf_bbc)
print(f"BBC News Categorization Accuracy: {accuracy_score(y_test_bbc, y_pred_bbc)}")
print("\nClassification Report for News Categorization:")
print(classification_report(y_test_bbc, y_pred_bbc))

# Save the BBC News Categorization Model and Vectorizer
joblib.dump(model_bbc, 'news_classifier_model.pkl')
joblib.dump(vectorizer_bbc, 'vectorizer_bbc.pkl')

# 2. Train a Fake News Detection Model
X_fake_news = fake_news_df_combined['processed_content']
y_fake_news = fake_news_df_combined['label']

X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(
    X_fake_news, y_fake_news, test_size=0.2, random_state=42
)

vectorizer_fake = TfidfVectorizer(max_features=5000)
X_train_tfidf_fake = vectorizer_fake.fit_transform(X_train_fake)
X_test_tfidf_fake = vectorizer_fake.transform(X_test_fake)

model_fake = LogisticRegression(max_iter=1000)
model_fake.fit(X_train_tfidf_fake, y_train_fake)

# Evaluate the Fake News Detection Model
y_pred_fake = model_fake.predict(X_test_tfidf_fake)
print(f"Fake News Detection Accuracy: {accuracy_score(y_test_fake, y_pred_fake)}")
print("\nClassification Report for Fake News Detection:")
print(classification_report(y_test_fake, y_pred_fake))

# Save the Fake News Detection Model and Vectorizer
joblib.dump(model_fake, 'fake_news_model.pkl')
joblib.dump(vectorizer_fake, 'vectorizer_fake.pkl')

# 3. Example of Using the Saved Models
# Load the Models and Vectorizers
model_bbc_loaded = joblib.load('news_classifier_model.pkl')
vectorizer_bbc_loaded = joblib.load('vectorizer_bbc.pkl')

model_fake_loaded = joblib.load('fake_news_model.pkl')
vectorizer_fake_loaded = joblib.load('vectorizer_fake.pkl')

# Example News Article for Categorization and Fake News Detection
sample_news = "Government announces a new policy to boost economy."

# Categorize the News
sample_vec_bbc = vectorizer_bbc_loaded.transform([preprocess_text(sample_news)])
predicted_category = model_bbc_loaded.predict(sample_vec_bbc)
print(f"News Category: {predicted_category[0]}")

# Detect if News is Real or Fake
sample_vec_fake = vectorizer_fake_loaded.transform([preprocess_text(sample_news)])
predicted_fake = model_fake_loaded.predict(sample_vec_fake)
print(f"Fake News Detection: {'Fake' if predicted_fake[0] == 'FAKE' else 'Real'}")
