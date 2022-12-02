import os
import tweepy
from dotenv import load_dotenv

# imports values from .env file
load_dotenv()

NUM_TWEETS = 100000

auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))

api = tweepy.API(auth)

def get_user_info(name):
    user = api.get_user(screen_name=name)
    return user

def Searchbykeyword(keyword):
    tweets = api.search_tweets(q=keyword, count=NUM_TWEETS)
    return tweets

def Searchbyhashtag(hashtag):
    tweets = api.search_tweets(q=hashtag, count=NUM_TWEETS)
    return tweets

def get_user_tweets(screen_name):
    # Gets tweets from user
    tweets = api.user_timeline(screen_name=screen_name, count=NUM_TWEETS, exclude_replies=True, include_rts=False)
    return tweets

def get_replies(tweet_id, screen_name):
    replies = []

    for reply in api.search_tweets(q='to:'+screen_name, count=NUM_TWEETS):
        if hasattr(reply, 'in_reply_to_status_id_str'):
            if (reply.in_reply_to_status_id==tweet_id):
                replies.append(reply) 
    return replies

def get_mentions(screen_name):
    mentions = []
    for mention in api.search_tweets(q='@'+screen_name, count=NUM_TWEETS):
        mentions.append(mention)
    return mentions

def tweet_processing(tweets):
    tweet_list = open("tweet_list.txt", "w", encoding="utf-8")
    for tweet in tweets:
        tweet_list.write(tweet.text)
        tweet_list.write("\n")
    tweet_list.close()
        
        
user = get_user_info("realDonaldTrump")