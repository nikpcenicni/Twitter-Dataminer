# Same function as Main.py, but with SVM instead of Naive Bayes as classifier
#import packages
#general purpose packages
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re, string, io, emoji, tweepy, csv, sys, nltk, os

#sklearn
from sklearn import preprocessing, metrics
from imblearn.over_sampling import RandomOverSampler 
from sklearn import svm #SVM
from sklearn.naive_bayes import BernoulliNB #Naive Bayes
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.model_selection import train_test_split #splitting data
from sklearn.feature_extraction.text import TfidfVectorizer#TFIDF vectorizer
from sklearn.feature_extraction.text import CountVectorizer #Count vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report #confusion matrix
from sklearn.linear_model import LogisticRegression, _logistic

from sklearn.utils import metaestimators

#XGBoost
from xgboost import XGBClassifier

#sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS


#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

import pickle

# first we retrieve data from training, test and validation set
# we use the same data as in the Naive Bayes classifier

#training set
df_train = pd.read_csv('Datasets/train.csv')
df_valid = pd.read_csv('Datasets/valid.csv')
df_test = pd.read_csv('Datasets/test.csv')

# since the data is already preprocessed, we can directly use the text column
# we also use the same features as in the Naive Bayes classifier

X_train = df_train['tweet_OG'].values
y_train_le = df_train['Sentiment'].values

X_valid = df_valid['tweet_OG'].values
y_valid = df_valid['Sentiment'].values

X_test = df_test['tweet_OG'].values
y_test_le = df_test['Sentiment'].values

clf = CountVectorizer()
X_train_cv = clf.fit_transform(X_train)
X_test_cv = clf.transform(X_test)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
X_train_tf = tf_transformer.transform(X_train_cv)
X_test_tf = tf_transformer.transform(X_test_cv)

# Next we can begin building the SVM classifier with rbf kernel
# we use the same parameters as in the Naive Bayes classifier
model = svm.SVC(kernel='rbf', C=1, gamma=1)
model.fit(X_train_tf, y_train_le)
model.score(X_test_tf, y_test_le)

# we can now predict the sentiment of the test set
y_pred = model.predict(X_test_tf)

# we can now evaluate the performance of the classifier
print(classification_report(y_test_le, y_pred))
print("accuracy: ", metrics.accuracy_score(y_test_le, y_pred))



