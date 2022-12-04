import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.util import pr
from nltk.corpus import stopwords
import string

def train_classifier():
    stemmer = nltk.SnowballStemmer('english')


    hate = pd.read_csv('Datasets/hatespeech.csv')
    hate['labels'] = hate['class'].map({0:'Hate Speech Detected', 1:'Offensive Language Detected', 2:'No Hate Speech Detected'})

    hate = hate[['tweet', 'labels']]
    hate.head()

    hate['tweet'] = hate['tweet'].apply(lambda x: clean_text(x))
    # print(hate.head())
        
    x = np.array(hate['tweet'])
    y = np.array(hate['labels'])

    cv = CountVectorizer()
    x = cv.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return cv, clf

def clean_text(text):
    stopword = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    return text


def classify(text, cv, clf):
    text = clean_text(text)
    text = cv.transform([text]).toarray()
    return clf.predict(text)[0]


def test():
    cv, clf = train_classifier()
    classification = classify("I hate you",cv, clf)
    print(classification)
    classification = classify("Kill yourself", cv, clf)
    print(classification)

# test()