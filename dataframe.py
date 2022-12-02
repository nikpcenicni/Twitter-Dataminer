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
import nltk

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
            analysis = TextBlob(tweet.full_text , analyzer=NaiveBayesAnalyzer())          
            w.writerow([tweet.created_at, 
                        tweet.full_text.replace('\n',' ').encode('utf-8'), 
                        tweet.user.screen_name.encode('utf-8'), 
                        [e['text'] for e in tweet._json['entities']['hashtags']],  
                        tweet.user.location, 
                        tweet.user.followers_count, 
                        tweet.retweet_count, 
                        tweet.favorite_count,
                        analysis.sentiment[0]])
        
        #perform sentiment analysis for each tweet in the dataset.csv file and add the results to the dataset in a new column
        # sentiment_analysis()

    
    #return file name
    return fname

def create_dataframe(fname):
    df = pd.read_csv('%s.csv' % fname)
    df.drop_duplicates(subset ="tweet_text", keep = False, inplace = True) # remove duplicate tweets
    # print(df.head())
    print(df.info)
    #change timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def main():
    #clear csv file
    open('dataset.csv', 'w').close()
    df = create_dataframe(create_CSV("covid19"))

if __name__ == "__main__":
    main()



