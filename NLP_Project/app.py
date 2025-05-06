from flask import Flask, request, jsonify, render_template
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process
import logging

# Initialize the Flask app
app = Flask(__name__)

# Load the saved models and vectorizers with error handling
def load_model(model_name):
    if not os.path.exists(model_name):
        raise FileNotFoundError(f'{model_name} not found')
    return joblib.load(model_name)

def load_fitted_vectorizer(file_path):
    vectorizer = joblib.load(file_path)
    try:
        vectorizer.transform(["test"])  # Test if vectorizer is fitted
    except ValueError:
        raise ValueError(f"The vectorizer loaded from {file_path} is not fitted.")
    return vectorizer

# Load models and vectorizers
try:
    news_classifier_model = load_model('news_classifier_model.pkl')
    vectorizer_bbc = load_fitted_vectorizer('vectorizer_bbc.pkl')
    fake_news_model = load_model('fake_news_model.pkl')
    vectorizer_fake = load_fitted_vectorizer('vectorizer_fake.pkl')
except (FileNotFoundError, ValueError) as e:
    print(e)
    exit(1)

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Function to extract keywords using TF-IDF
def extract_keywords(text, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()
    keywords = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    return [keyword[0] for keyword in sorted_keywords]

# Route for the homepage (Frontend)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle POST requests for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data or 'task' not in data:
        return jsonify({'error': 'Text and task type are required'}), 400

    text = data['text']
    task = data['task']
    processed_text = preprocess_text(text)

    if task == "categorization":
        try:
            vectorized_text = vectorizer_bbc.transform([processed_text])
            prediction = news_classifier_model.predict(vectorized_text)
            return jsonify({'task': 'categorization', 'category': prediction[0]})
        except Exception as e:
            return jsonify({'error': f'Categorization failed: {str(e)}'}), 500

    elif task == "fake_news":
        try:
            vectorized_text = vectorizer_fake.transform([processed_text])
            prediction = fake_news_model.predict(vectorized_text)
            fake_or_real = "FAKE" if prediction[0] == "FAKE" else "REAL"
            return jsonify({'task': 'fake_news', 'result': fake_or_real})
        except Exception as e:
            return jsonify({'error': f'Fake news detection failed: {str(e)}'}), 500

    elif task == "keywords":
        try:
            keywords = extract_keywords(processed_text)
            return jsonify({'task': 'keywords', 'keywords': keywords})
        except Exception as e:
            return jsonify({'error': f'Keyword extraction failed: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Invalid task type. Use "categorization", "fake_news", or "keywords"'}), 400

# Knowledge Base of FAQs (Storing responses in a dictionary)
faq_knowledge_base = {
    "classification": "News classification categorizes articles into predefined topics based on their content.",
    "fake news detection": "Fake news detection identifies whether news is real or fake using advanced ML models.",
    "fake news": "Fake news detection identifies whether news is real or fake using advanced ML models.",
    "algorithm": "We use Logistic Regression and TF-IDF vectorization for text classification, fake news detection, and keywords extraction.",
    "categories": "Our categorization system supports business, entertainment, politics, sport, and tech.",
    "accuracy": "The accuracy depends on the model and dataset. We aim for high accuracy by training with diverse data.",
    "tf-idf": "TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numeric features for ML models.",
    "keyword": "Keyword extraction identifies important words or phrases from text using techniques like TF-IDF."
}

# Function to find FAQ response based on keywords
def get_faq_response(user_message):
    user_message = user_message.lower()
    logging.debug(f"User message: {user_message}")
    keywords = list(faq_knowledge_base.keys())
    
    # Use fuzzy matching to find the best match for the user's message
    best_match, score = process.extractOne(user_message, keywords)
    if score > 70:  # Adjust threshold as needed
        logging.debug(f"Best match: {best_match} with score: {score}")
        return faq_knowledge_base[best_match]
    
    return None  # Return None when no FAQ match is found

# Function to handle basic greetings and other small talk
def get_basic_response(user_message):
    user_message = user_message.lower()
    if any(greeting in user_message for greeting in ["hello", "hi"]):
        return "Hello! How can I assist you?"
    elif any(farewell in user_message for farewell in ["goodbye", "bye"]):
        return "Goodbye! Have a great day!"
    elif any(thank in user_message for thank in ["thanks", "thank you"]):
        return "You're welcome! Let me know if you need further assistance."
    return "I don't quite understand. Ask about news classification, fake news detection, or keyword extraction, and I'll assist you!"

# Route to handle chatbot responses
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_message = data['message']

    # Try to get the FAQ response first
    bot_response = get_faq_response(user_message)
    if bot_response:
        return jsonify({'response': bot_response})

    # Check for basic responses as a fallback
    basic_response = get_basic_response(user_message)
    return jsonify({'response': basic_response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
