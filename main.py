from analyzer import *
from twitter import *


def replys():
    pass

def mentions():
    pass


def operations(screen_name):
    operation = input("Enter 1 to analyze tweet replies, 2 to analyze mentions, or any other value to quit: ")
    if operation == '1':
        tweet_id = input("Enter tweet id: ")
        tweets = get_replies(tweet_id, screen_name)
        return tweets
    elif operation == '2':
        tweets = get_mentions(screen_name)
        return tweets
    else:
        return None


def main():
    screen_name = input("Enter a Twitter handle: ")
    # if screen_name[0] == '@':
    #     screen_name = screen_name[1:]
        
    user = get_user_info(screen_name)
    
    print("User: " + user.name)
    print("Number of Followers: " + str(user.followers_count))
    print("Number of Friends: " + str(user.friends_count))
    print("Number of Tweets: " + str(user.statuses_count))
    print("Number of Likes: " + str(user.favourites_count))
    
    tweets = operations(screen_name)
    if tweets != None:
        tweet_processing(tweets)
        print("Tweets processed")
        print("Analyzing tweets...")
        emotions = preprocess_text()
        print(f"Emotional associations: {emotions}")
        print(sentiment_analysis(clean_tweets()))

    # tweet_processing(tweets)

    # # select the tweet you want to analyze
    
    # # Get the sentiment of the tweets
    
    # cleaned_tweets = clean_tweets()
    # sentiment = sentiment_analysis(cleaned_tweets)

    # # Print the sentiment
    # print(sentiment)
    
main()