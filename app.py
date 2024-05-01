from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the CSV file
new_data = pd.read_csv('csv_files/train_v2_drcat_02.csv')

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_data['text'])

# Naive Bayes

data = pd.read_csv('csv_files/train_v2_drcat_02.csv')

X = tfidf_matrix  # Use the TF-IDF matrix obtained earlier
y = data['generated']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict function
def predict(text):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
    prediction = nb_classifier.predict(tfidf_vector)
    return "AI-generated" if prediction == 1 else "Human-written"

# Flask routes
@app.route('/')
def index():
    return render_template('file1.html')

@app.route('/predict', methods=['POST'])
def predict_text():
    text = request.form['text']
    prediction = predict(text)
    return render_template('file1.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
