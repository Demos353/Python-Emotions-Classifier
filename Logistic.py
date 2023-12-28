import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    # Stemming and Lemmatization
    stemmed = [stemmer.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return " ".join(lemmatized)

def train_logistic_regression(data_file):
    data = pd.read_csv(data_file)
    data['Clean_Text'] = data['Comment'].apply(preprocess_text)

    X = data['Clean_Text']
    y = data['Emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X_train)

    logreg_classifier = LogisticRegression(max_iter=1000)
    logreg_classifier.fit(X_tfidf, y_train)

    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    logreg_predictions = logreg_classifier.predict(X_test_tfidf)

    logreg_accuracy = accuracy_score(y_test, logreg_predictions)
    return logreg_accuracy,logreg_classifier,tfidf_vectorizer

def classify_logistic_regression(text, classifier, vectorizer):
    text_clean = preprocess_text(text)
    text_tfidf = vectorizer.transform([text_clean])
    prediction = classifier.predict(text_tfidf)
    return prediction[0]


accuracy, classifier, vectorizer = train_logistic_regression("Emotion_classify_Data.csv")

prediction_angry = classify_logistic_regression("shut up ,I'm too tired of hearing this crap",classifier,vectorizer)
prediction_fear = classify_logistic_regression("This is so scary , dont show me this video again ",classifier,vectorizer)
prediction_joy = classify_logistic_regression("you Made my day <3",classifier,vectorizer)
print(accuracy)
print(prediction_angry)
print(prediction_fear)
print(prediction_joy)