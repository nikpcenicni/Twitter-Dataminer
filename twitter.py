import os
import time
import tweepy
from dotenv import load_dotenv

# imports values from .env file
load_dotenv()

NUM_TWEETS = 1000

auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))

api = tweepy.API(auth)

# Get the User object for twitter...
user = api.get_user(screen_name='JustinTrudeau')
print(user.screen_name)
print(user.followers_count)
print(user.id)

# Gets tweets from user
tweets = api.user_timeline(screen_name='JustinTrudeau', count=NUM_TWEETS, exclude_replies=True, include_rts=False)
for tweet in tweets:
    print(tweet.text)
    
name = user.screen_name
tweet_id = tweets[1].id


def limit_handled(cursor):
    while True:
        try:
            yield next(cursor)
        except tweepy.RateLimitError:
            time.sleep(15 * 60)

def get_replies(tweet_id, name):
    replies = []

    for reply in api.search_tweets(q='to:'+name, count=NUM_TWEETS):
        if hasattr(reply, 'in_reply_to_status_id_str'):
            if (reply.in_reply_to_status_id==tweet_id):
                replies.append(reply) 
    return replies

def get_mentions(name):
    mentions = []
    for mention in api.search_tweets(q='@'+name, count=NUM_TWEETS):
        if "RT @" not in mention.text:
            mentions.append(mention)
    return mentions

def tweet_processing(tweets):
    tweet_list = open("tweet_list.txt", "a")
    for tweet in tweets:
        tweet_list.write(tweet.text)
        tweet_list.write("\n")
    tweet_list.close()
    
tweet_processing(get_mentions(name))