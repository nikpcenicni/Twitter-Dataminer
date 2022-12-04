import tweepy
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

NUM_TWEETS = 10000

auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))

api = tweepy.API(auth, wait_on_rate_limit=True)

def get_tweets(keyword):
    filename="Datasets/tweets.csv"
    if (not os.path.exists(filename)):
        df = pd.DataFrame({'timestamp': [],
                   'tweet_OG': [],
                   'username': [],
                   'all_hashtags': [],
                   'location': [],
                   'followers_count': [],
                   'retweet_count': [],
                   'favorite_count': [],
                   'Sentiment': []})
    else:
        df = pd.read_csv(filename)
    #search for tweets with the hashtag or keyword = 'WorldCup'
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang='en', tweet_mode='extended').items(NUM_TWEETS):
        # timestamp,tweet_OG,username,all_hashtags,location,followers_count,retweet_count,favorite_count,Sentiment
        # df.append(new_row, tweet.full_text)
        df.loc[len(df.index)] = [tweet.created_at, tweet.full_text, tweet.user.screen_name, tweet.entities['hashtags'], tweet.user.location, tweet.user.followers_count, tweet.retweet_count, tweet.favorite_count, '']
    df.to_csv(filename,index=False, encoding='utf-8')
        
get_tweets("World Cup")