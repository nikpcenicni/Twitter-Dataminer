import os
import time
import tweepy
from dotenv import load_dotenv

# imports values from .env file
load_dotenv()

auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))

api = tweepy.API(auth)

# Get the User object for twitter...
user = api.get_user(screen_name='ElonMusk')
print(user.screen_name)
print(user.followers_count)
print(user.id)

# Gets tweets from user
tweets = api.user_timeline(screen_name='ElonMusk', count=50, exclude_replies=True)
for tweet in tweets:
    print(tweet.text)
    
name = user.screen_name
tweet_id= tweets[0].id


def limit_handled(cursor):
    while True:
        try:
            yield next(cursor)
        except tweepy.RateLimitError:
            time.sleep(15 * 60)
            
            
## Functions below broken rn, syntax changed in update of tweepy which broke this code that worked before 4.0
                
# def get_replies(tweet_id, name):
#     replies = []
#     for reply in tweepy.Cursor(api.search_recent_tweets, q='to:'+name, timeout=999999).items(100):
#         if hasattr(reply, 'in_reply_to_status_id_str'):
#             if (reply.in_reply_to_status_id_str==tweet_id):
#                 replies.append(reply)
#     return replies

# replies = get_replies(tweet_id, name)
# for reply in replies:
#     print(reply.text)