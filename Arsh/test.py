# program to scrape tweets from twitter using the tweepy library
# tweets can be scraped using a hashtag, a keyword or a username
# mined tweets are stored in a csv file and then tweets are cleaned and preprocessed
# then the dataset is stemmed and lemmatized
# the cleaned tweets are then used to train a sentiment analysis model
# The bertTokenizerFast is used to tokenize tweets with refenence to positive and negative words located in the text file: positive-words.txt and negative-words.txt
# tweets are then classified as extreme positive, positive, neutral, negative, extreme negative
# then the data is split into training and testing subsets abd transformed using TFIDF vectorizer where the training data is 70 percent of the dataset and the testing data is 30 percent of the dataset
# The trained data is then evaluated with testing data to check what accuracy is generated accuracy score, ROC curve and f1 score
# The accuracy is then explained with the help of the Confusion Matrix.
# Then classification is performed using Bernoulli Naive Bayes, SVM Classifier XGBoost Classifier (random forest)
# the results are then compared using the the accuracy score, ROC curve and f1 score 
# finally, the best model is selected and used to predict the sentiment of tweets scraped from twitter using the tweepy library


#import packages
#general purpose packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re, string, io, emoji, tweepy, csv, sys, nltk, os

#sklearn
from sklearn import preprocessing, metrics
from imblearn.over_sampling import RandomOverSampler 
from sklearn.svm import LinearSVC #SVM
from sklearn.naive_bayes import BernoulliNB #Naive Bayes
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.model_selection import train_test_split #splitting data
from sklearn.feature_extraction.text import TfidfVectorizer #TFIDF vectorizer
from sklearn.feature_extraction.text import CountVectorizer #Count vectorizer
from sklearn.metrics import confusion_matrix, classification_report #confusion matrix

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

#keras
import tensorflow as tf
from tensorflow import keras

#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


#extract tweets from twitter using tweepy and store in csv file
def create_CSV():
    #connect to twitter api
    from dotenv import load_dotenv
    load_dotenv
    auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))
    api = tweepy.API(auth, wait_on_rate_limit=True)

    #name of csv file to be created
    fname = "dataset.csv"

    #open csv file
    with open(fname, 'w', encoding='utf-8') as file:
        W = csv.writer(file)

        #write header row to csv file
        W.writerow(['timestamp', 'tweet_OG', 'username', 'all_hashtags', 'location', 
                    'followers_count', 'retweet_count', 'favorite_count', 'Sentiment'])

        #search for tweets with the hashtag or keyword = 'WorldCup'
        for tweet in tweepy.Cursor(api.search, q='#WorldCup', lang='en', tweet_mode='extended').items(1000):
            W.writerow([tweet.created_at, tweet.full_text.replace('\n',' ').encode('utf-8'), 
                        tweet.user.screen_name, [e['text'] for e in tweet._json['entities']['hashtags']], 
                        tweet.user.location, tweet.user.followers_count, tweet.retweet_count, tweet.favorite_count, ''])


    #close csv file
    file.close()

#clean data 
def clean_data():
    df = pd.read_csv('dataset.csv')
    df.drop_duplicates(subset ="tweet_OG", keep = False, inplace = True) # remove duplicate tweets
    df['timestamp'] = pd.to_datetime(df['timestamp'])     #change timestamp to datetime

    #call clean_twwet function
    df['tweet_OG'] = df['tweet_OG'].apply(clean_tweet)
    
    #remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    df['tweet_OG'] = df['tweet_OG'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    #remove empty tweets
    df = df[df['tweet_OG'] != '']

    #get tokenized version of tweets using bertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    df['tokenized'] = df['tweet_OG'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    #drop tokenized tweets with more than 80 tokens (not english)
    df = df[df['tokenized'].apply(lambda x: len(x) <= 80)]
    #replace original tweet with tokenized tweet
    df['tweet_OG'] = df['tokenized'].apply(lambda x: tokenizer.decode(x))

    #perform lemmatization and stemming
    df['tweet_OG'] = df['tweet_OG'].apply(lemmatize)
    df['tweet_OG'] = df['tweet_OG'].apply(stem)
    
    #perform sentiment analysis on tweets
    df['Sentiment'] = df['tweet_OG'].apply(get_sentiment)

    return df

def get_sentiment(tweet):
    #get sentiment using vaderSentiment and classify tweets as extreme positive, positive, neutral, negative, extreme negative
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(tweet)
    if ss['compound'] >= 0.05:
        return 'Extreme Positive'
    elif ss['compound'] > -0.05 and ss['compound'] < 0.05:
        return 'Neutral'
    else:
        return 'Extreme Negative'
    

def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def stem(text):
    stemmer = nltk.stem.PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

def clean_tweet(tweet):
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet) #remove links and mentions
    tweet = re.sub(r'[^\x00-\x7f]',r'', tweet) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    tweet = tweet.translate(table)
    #remove hashtags
    tweet = re.sub(r'#\S+', '', tweet)
    #remove mentions
    tweet = re.sub(r'@\S+', '', tweet)
    #remove emojis
    tweet = emoji.demojize(tweet)
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')
    #remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    #remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    #remove multiple spaces
    tweet = re.sub(r'\s+', ' ', tweet)
    #remove leading and trailing spaces
    tweet = tweet.strip()
    #convert to lowercase
    tweet = tweet.lower()
    return tweet

def main():
    # create_CSV()
    df = clean_data()
    df.info()
    print(df.head())

if __name__ == "__main__":
    main()





    








