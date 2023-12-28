import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def train_naive_bayes(data_file):
    data = pd.read_csv(data_file)
    X = data['Comment']
    y = data['Emotion']
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(X)
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_count, y)
    return nb_classifier, count_vectorizer

def classify_naive_bayes(text, classifier, vectorizer):
    text_count = vectorizer.transform([text])
    prediction = classifier.predict(text_count)
    return prediction[0]

classifier, vectorizer = train_naive_bayes("Emotion_classify_Data.csv")

prediction_angry = classify_naive_bayes("shut up ,I'm too tired of hearing this crap",classifier,vectorizer)
prediction_fear = classify_naive_bayes("This is so scary , dont show me this video again ",classifier,vectorizer)
prediction_joy = classify_naive_bayes("you Made my day <3",classifier,vectorizer)

print(prediction_angry)
print(prediction_fear)
print(prediction_joy)

