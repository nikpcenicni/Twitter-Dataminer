import os
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

   