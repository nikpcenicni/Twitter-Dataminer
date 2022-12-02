#TO-DO:
# Implement Naive Bayes Classifier
# Implement Logistic Regression
# Implement SVM


#general purpose packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#data processing
import os, re, string, io, emoji, tweepy, csv, sys

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn import preprocessing, metrics
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

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
        w.writerow(['timestamp', 'tweet_OG', 'username', 'all_hashtags', 'location', 
                    'followers_count', 'retweet_count', 'favorite_count', 'Sentiment'])
       
        # For each tweet matching hashtag and write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.search_tweets, q=hashtag_phrase+' -filter:retweets', lang="en", tweet_mode='extended').items(50):
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
                        sentiment]) 
    #return file name
    return fname

def create_dataframe(fname):
    df = pd.read_csv('%s.csv' % fname)
    df.drop_duplicates(subset ="tweet_OG", keep = False, inplace = True) # remove duplicate tweets
    df['timestamp'] = pd.to_datetime(df['timestamp'])     #change timestamp to datetime

    #clean tweets
    df['tweet_OG'] = df['tweet_OG'].apply(clean_tweet)

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    df['tweet_OG'] = df['tweet_OG'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    #lemmatize
    lemmatizer = WordNetLemmatizer()
    df['tweet_OG'] = df['tweet_OG'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    #stemming
    stemmer = PorterStemmer()
    df['tweet_OG'] = df['tweet_OG'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

    #drop tweets with with less than 3 words
    df = df[df['tweet_OG'].str.split().str.len() > 3]

    #remove tweets with less than 3 characters
    df = df[df['tweet_OG'].str.len() > 3]

    #shuffle dataframe and reset index
    df = df.sample(frac=1).reset_index(drop=True)

    #tokenize with BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    df['tweet_OG'] = df['tweet_OG'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    #create test set
    df_test = df.sample(frac=0.2, random_state=0)
    df_test = df_test.reset_index(drop=True)

    #balance classes in dataset with RandomOverSampler
    #oversample the train test to remove bias towards the majority classes.
    ros = RandomOverSampler(random_state=0)
    x_train, y_train = ros.fit_resample(df['tweet_OG'].values.reshape(-1,1), df['Sentiment'].values.reshape(-1,1))
    train_os = pd.DataFrame(list(zip([x[0] for x in x_train], y_train)), columns = ['tweet_OG', 'Sentiment']);

    x = train_os['tweet_OG'].values
    y = train_os['Sentiment'].values

    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(df['tweet_OG'], df['Sentiment'], test_size=0.1, random_state=42)

    x_test = df_test['tweet_OG'].values
    y_test = df_test['Sentiment'].values

    #create copies of the test set
    y_train_le = y_train.copy()
    y_test_le = y_test.copy()

    #Improve accuracy of model by encoding labels with one hot encoding
    ohe = preprocessing.OneHotEncoder()
    y_train = ohe.fit_transform(y_train.values.reshape(-1,1)).toarray()
    y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()

    #call function to perform simple naive bayes classification with baseline accuracy
    naive_bayes_baseline(X_train, X_test, y_train_le, y_test_le)
    return df

def naive_bayes_baseline(X_train, X_test, y_train, y_test):
    #token count vectorizer
    vectorizer = CountVectorizer()
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    #train model
    clf = MultinomialNB().fit(X_train_tf, y_train)

    #predict
    predicted = clf.predict(X_test_tf)

    #print accuracy
    print("Naive Bayes Accuracy: ", accuracy_score(y_test, predicted))

    #print classification report
    print(classification_report(y_test, predicted))


def accuracy_score(y_test, y_pred):
    return np.sum(y_test == y_pred, axis=0) / y_test.shape[0]

def classification_report(y_test, y_pred):
    target_names = ['negative', 'neutral', 'positive']
    return metrics.classification_report(y_test, y_pred, target_names=target_names)


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
    create_CSV("WorldCup")
    # df = create_dataframe(create_CSV("covid19"))
    # df.info()


if __name__ == "__main__":
    main()



