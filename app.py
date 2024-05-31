from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download NLTK resources
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# Load the models and vectorizer
nb_classifier = joblib.load('naive_bayes_model.pkl')
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Print model details
print("Loaded Naive Bayes Classifier:", nb_classifier)
print("Loaded Logistic Regression Model:", logistic_regression_model)
print("Logistic Regression Intercept:", logistic_regression_model.intercept_)
print("Logistic Regression Coefficients:", logistic_regression_model.coef_)
print("Loaded TF-IDF Vectorizer:", tfidf_vectorizer)

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

# Predict NB function
def predictNB(text):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
    prediction = nb_classifier.predict(tfidf_vector.toarray())
    return "AI-generated" if prediction == 1 else "Human-written"

# Predict LR function
def predictLR(text):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
    prediction = logistic_regression_model.predict(tfidf_vector)
    return "AI-generated" if prediction == 1 else "Human-written"

# Flask routes
@app.route('/')
def index():
    return render_template('essai.html')

@app.route('/predict', methods=['POST'])
def predict_text():
    try:
        text = request.form['text']
        model = request.form['model']
        if model == 'Naive_Bayes':
            prediction = predictNB(text)
        elif model == 'Logistic_Regression':
            prediction = predictLR(text)
        return render_template('essai.html', prediction=f"The {model.replace('_', ' ')} model predicted: {prediction}")
    except Exception as e:
        print(e)
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
