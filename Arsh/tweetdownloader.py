import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

NUM_TWEETS = 100000

auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))

api = tweepy.API(auth, wait_on_rate_limit=True)

def get_tweets():
    #connect to twitter api
    from dotenv import load_dotenv
    load_dotenv
    #search for tweets with the hashtag or keyword = 'WorldCup'
    for tweet in tweepy.Cursor(api.search, q='#WorldCup', lang='en', tweet_mode='extended').items(1000):
        print(tweet.full_text)
        
        
get_tweets()