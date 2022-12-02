#general purpose packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import os, re, string, io
import json
import csv
import tweepy

#data processing
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#sentiment analysis
from textblob import TextBlob
# Importing the NaiveBayesAnalyzer classifier from NLTK
from textblob.sentiments import NaiveBayesAnalyzer

#keras
import tensorflow as tf
from tensorflow import keras

# import vaderSentiment as vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

def create_CSV(hashtag_phrase):

    #setup twitter api
    NUM_TWEETS = 10000
    # imports values from .env file
    from dotenv import load_dotenv
    load_dotenv()
    # Initializing Tweepy API
    auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))
    api = tweepy.API(auth, wait_on_rate_limit=True)
    # Name of csv file to be created
    fname = "dataset"
    
    # Open the spreadsheet
    with open('%s.csv' % (fname), 'w', encoding="utf-8") as file:
        w = csv.writer(file)
        
        # Write header row (feature column names of your choice)
        w.writerow(['timestamp', 'tweet_text', 'username', 'all_hashtags', 'location', 
                    'followers_count', 'retweet_count', 'favorite_count', 'Sentiment'])
       
        # For each tweet matching hashtag and write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.search_tweets, q=hashtag_phrase+' -filter:retweets', lang="en", tweet_mode='extended').items(10):
            #perform sentiment analysis with vadarsentiment
            sentiment = SentimentIntensityAnalyzer().polarity_scores(tweet.full_text).get('compound')
            #get value of sentiment
            if sentiment >= 0.05:
                sentiment = 'positive'
            elif sentiment <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            # analysis = TextBlob(tweet.full_text , analyzer=NaiveBayesAnalyzer())          
            w.writerow([tweet.created_at, 
                        tweet.full_text.replace('\n',' ').encode('utf-8'), 
                        tweet.user.screen_name.encode('utf-8'), 
                        [e['text'] for e in tweet._json['entities']['hashtags']],  
                        tweet.user.location, 
                        tweet.user.followers_count, 
                        tweet.retweet_count, 
                        tweet.favorite_count,
                        sentiment['compound']])
        
        #perform sentiment analysis for each tweet in the dataset.csv file and add the results to the dataset in a new column
        # sentiment_analysis()

    
    #return file name
    return fname

def create_dataframe(fname):
    df = pd.read_csv('%s.csv' % fname)
    df.drop_duplicates(subset ="tweet_text", keep = False, inplace = True) # remove duplicate tweets
    df['timestamp'] = pd.to_datetime(df['timestamp'])     #change timestamp to datetime

    #clean tweets
    df['tweet_text'] = df['tweet_text'].apply(clean_tweet)

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    #lemmatize
    lemmatizer = WordNetLemmatizer()
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    #stemming
    stemmer = PorterStemmer()
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

    #tokenize with BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    df['tweet_text'] = df['tweet_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    return df

# clean the tweets
def clean_tweet(tweet):
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet) #remove links and mentions
    tweet = re.sub(r'[^\x00-\x7f]',r'', tweet) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    # #remove links    # tweet = re.sub(r'http\S+', '', tweet)
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
    #clear csv file
    open('dataset.csv', 'w').close()
    df = create_dataframe(create_CSV("covid19"))
    df.info()

if __name__ == "__main__":
    main()



