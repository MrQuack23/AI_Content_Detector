import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

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

# Apply preprocessing to the entire dataset
new_data['text'] = new_data['text'].apply(preprocess_text)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(new_data['text'])  # for train test split

# Labels
y = new_data['generated']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(new_data.head())

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Determine the number of batches
n_samples = X_train.shape[0]
batch_size = 1000  # Adjust this value based on your memory capacity
n_batches = n_samples // batch_size

# Train the classifier in batches
for batch in gen_batches(n_samples, batch_size):
    nb_classifier.partial_fit(X_train[batch].toarray(), y_train[batch], np.unique(y_train))

# Save the Naive Bayes model
joblib.dump(nb_classifier, 'naive_bayes_model.pkl')

# Initialize and train the Logistic Regression classifier
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Save the Logistic Regression model
joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(X, 'tfidf_vectorizer.pkl')


