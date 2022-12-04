import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer('english')
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

hate = pd.read_csv('hatespeech.csv')
hate['labels'] = hate['class'].map({0:'Hate Speech Detected', 1:'Offensive Language Detected', 3:'No Hate Speech Detected'})

hate = hate[['tweet', 'labels']]
hate.head()



def clean_text(text):
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

hate['tweet'] = hate['tweet'].apply(lambda x: clean_text(x))
# print(hate.head())
print(hate.head())
    
x = np.array(hate['tweet'])
y = np.array(hate['labels'])

cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
# print (clf.score(x_test, y_test))
# test = ['I hate you']

# hate = cv.transform([test]).toarray()
# print(clf.predict(hate))
 