# program to scrape tweets from twitter using the tweepy library
# tweets can be scraped using a hashtag, a keyword or a username
# mined tweets are stored in a csv file and then tweets are cleaned and preprocessed
# then the dataset is stemmed and lemmatized
# the cleaned tweets are then used to train a sentiment analysis model
# The bertTokenizerFast is used to tokenize tweets with refenence to positive and negative words located in the text file: positive-words.txt and negative-words.txt
# tweets are then classified as extreme positive, positive, neutral, negative, extreme negative
# then the data is split into training and testing subsets abd transformed using TFIDF vectorizer where the training data is 70 percent of the dataset and the testing data is 30 percent of the dataset
# The trained data is then evaluated with testing data to check what accuracy is generated accuracy score, ROC curve and f1 score
# The accuracy is then explained with the help of the Confusion Matrix.
# Then classification is performed using Bernoulli Naive Bayes, SVM Classifier XGBoost Classifier (random forest)
# the results are then compared using the the accuracy score, ROC curve and f1 score 
# finally, the best model is selected and used to predict the sentiment of tweets scraped from twitter using the tweepy library


#import packages
#general purpose packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re, string, io, emoji, tweepy, csv, sys, nltk, os

#sklearn
from sklearn import preprocessing, metrics
from imblearn.over_sampling import RandomOverSampler 
from sklearn.svm import LinearSVC #SVM
from sklearn.naive_bayes import BernoulliNB #Naive Bayes
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.model_selection import train_test_split #splitting data
from sklearn.feature_extraction.text import TfidfVectorizer#TFIDF vectorizer
from sklearn.feature_extraction.text import CountVectorizer #Count vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report #confusion matrix

#XGBoost
from xgboost import XGBClassifier

#sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS


#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

#keras
import tensorflow as tf
from tensorflow import keras

#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


#extract tweets from twitter using tweepy and store in csv file
def create_CSV():
    #connect to twitter api
    from dotenv import load_dotenv
    load_dotenv
    auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))
    api = tweepy.API(auth, wait_on_rate_limit=True)

    #name of csv file to be created
    fname = "dataset.csv"

    #open csv file
    with open(fname, 'w', encoding='utf-8') as file:
        W = csv.writer(file)

        #write header row to csv file
        W.writerow(['timestamp', 'tweet_OG', 'username', 'all_hashtags', 'location', 
                    'followers_count', 'retweet_count', 'favorite_count', 'Sentiment'])

        #search for tweets with the hashtag or keyword = 'WorldCup'
        for tweet in tweepy.Cursor(api.search, q='#WorldCup', lang='en', tweet_mode='extended').items(1000):
            W.writerow([tweet.created_at, tweet.full_text.replace('\n',' ').encode('utf-8'), 
                        tweet.user.screen_name, [e['text'] for e in tweet._json['entities']['hashtags']], 
                        tweet.user.location, tweet.user.followers_count, tweet.retweet_count, tweet.favorite_count, ''])


    #close csv file
    file.close()

#clean data 
def clean_data():
    df = pd.read_csv('dataset.csv')
    df.drop_duplicates(subset ="tweet_OG", keep = False, inplace = True) # remove duplicate tweets
    df['timestamp'] = pd.to_datetime(df['timestamp'])     #change timestamp to datetime

    #call clean_twwet function
    df['tweet_OG'] = df['tweet_OG'].apply(clean_tweet)
    
    #remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    df['tweet_OG'] = df['tweet_OG'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    #remove empty tweets
    df = df[df['tweet_OG'] != '']

    #get tokenized version of tweets using bertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    df['tokenized'] = df['tweet_OG'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    #drop tokenized tweets with more than 80 tokens (not english)
    df = df[df['tokenized'].apply(lambda x: len(x) <= 80)]
    #replace original tweet with tokenized tweet
    df['tweet_OG'] = df['tokenized'].apply(lambda x: tokenizer.decode(x))

    #perform lemmatization and stemming
    df['tweet_OG'] = df['tweet_OG'].apply(lemmatize)
    df['tweet_OG'] = df['tweet_OG'].apply(stem)
    
    #perform sentiment analysis on tweets
    df['Sentiment'] = df['tweet_OG'].apply(get_sentiment)

    #shuffle dataset and reset index
    df['Sentiment'] = df['Sentiment'].map({'Extremely Negative':0,'Negative':0,'Neutral':1,'Positive':2,'Extremely Positive':2})
    df = df.sample(frac=1).reset_index(drop=True)
    df_test = df.copy()

    #crossbalancing dataset using RandomOverSampler to create training x and y
    ros = RandomOverSampler(random_state=0)
    train_x, train_y = ros.fit_resample(np.array(df['tweet_OG']).reshape(-1, 1), np.array(df['Sentiment']).reshape(-1, 1))
    train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['tweet_OG', 'Sentiment']);

    #split dataset into train, test and validation sets
    X = train_os['tweet_OG']
    y = train_os['Sentiment']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_test = df_test['tweet_OG'].values
    y_test = df_test['Sentiment'].values

    
    #create copies of train, test and validation sets
    y_train_le = X_train.copy()
    y_test_le = X_test.copy()
    y_valid_le = X_valid.copy()

    #encode with one hot encoding
    ohe = preprocessing.OneHotEncoder()
    y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
    y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()
    y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()

    print(f"TRAINING DATA: {X_train.shape[0]}\nVALIDATION DATA: {X_valid.shape[0]}\nTESTING DATA: {X_test.shape[0]}" )
    print()

    #call naive_bayes function
    naive_bayes(X_train, X_test, y_train_le, y_test_le)
    return df

def naive_bayes(X_train, X_test, y_train_le, y_test_le):
    clf = CountVectorizer()
    X_train_cv = clf.fit_transform(X_train)
    X_test_cv = clf.transform(X_test)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)

    nb_classifier = BernoulliNB()
    nb_classifier.fit(X_train_tf, y_train_le)

    nb_predictions = nb_classifier.predict(X_test_tf)

    print("Naive Bayes Accuracy: ", accuracy_score(y_test_le, nb_predictions))
    print()
    #classification report
    print(classification_report(y_test_le, nb_predictions, target_names=['Negative', 'Neutral', 'Positive']))
    print()


def get_sentiment(tweet):
    #get sentiment using vaderSentiment and classify tweets as extreme positive, positive, neutral, negative, extreme negative
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(tweet)
    if ss['compound'] >= 0.05:
        return 'Extremely Positive'
    elif ss['compound'] > -0.05 and ss['compound'] < 0.05:
        return 'Neutral'
    else:
        return 'Extremely Negative'
    

def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def stem(text):
    stemmer = nltk.stem.PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

def clean_tweet(tweet):
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet) #remove links and mentions
    tweet = re.sub(r'[^\x00-\x7f]',r'', tweet) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
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
    # create_CSV()
    df = clean_data()
    # df.info()
    # print(df.head())

if __name__ == "__main__":
    main()





    








