from analyzer import *
from twitter import *

import os


def reset():
    open("tweet_list.txt", "w").close()


def replys(screen_name):
    
    tweets = get_user_tweets(screen_name)
    i = 0
    for tweet in tweets:
        print(f"{i}: {tweet.text}")
        i+=1
        
    id = int(input("Enter the number of the tweet you want to analyze: "))
    tweet_id = tweets[id].id
    
    tweets = get_replies(tweet_id, screen_name)
    
    return tweets
    # = get_replies(tweet_id, screen_name)
    

def mentions(screen_name):
    tweets = get_mentions(screen_name)
    return tweets
    pass


def operations(screen_name):
    operation = input("Enter 1 to analyze tweet replies, 2 to analyze mentions, or any other value to quit: ")
    if operation == '1':
        tweets =  replys(screen_name)
        return tweets
    elif operation == '2':
        tweets = mentions(screen_name)
        return tweets
    else:
        return None

def user_information():
    screen_name = input("Enter a Twitter handle: ")
    # if screen_name[0] == '@':
    #     screen_name = screen_name[1:]
        
    user = get_user_info(screen_name)
    
    print("User: " + user.name)
    print("Number of Followers: " + str(user.followers_count))
    print("Number of Friends: " + str(user.friends_count))
    print("Number of Tweets: " + str(user.statuses_count))
    print("Number of Likes: " + str(user.favourites_count))
    return screen_name

def show_graphs(emotions):
    graph_emotions(emotions)

def main():
    
    screen_name = user_information()
    
    tweets = operations(screen_name)
    if tweets != None:
        tweet_processing(tweets)
        print("Tweets processed")
        print("Analyzing tweets...")
        emotions = preprocess_text()
        print(f"Emotional associations: {emotions}")
        print(sentiment_analysis(clean_tweets()))
        option = input("Enter 1 to show graphs, or any other value to quit: ")
        if option == '1':
            show_graphs(emotions)
    
    reset()
    
main()