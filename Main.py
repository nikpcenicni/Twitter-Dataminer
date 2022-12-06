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

# then the negative tweets are stored before hate speech detection is performed
# hate speech detection is performed using a deep learning model

# the model is then used to predict the sentiment of the tweets
# the tweets are then classified as hate speech, offensive language, neither


#import packages
#general purpose packages
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re, string, io, emoji, tweepy, csv, sys, nltk, os, time

#sklearn
from sklearn import preprocessing, metrics
from imblearn.over_sampling import RandomOverSampler 
from sklearn import svm #SVM
from sklearn.naive_bayes import BernoulliNB #Naive Bayes
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.model_selection import train_test_split #splitting data
from sklearn.feature_extraction.text import TfidfVectorizer#TFIDF vectorizer
from sklearn.feature_extraction.text import CountVectorizer #Count vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve #confusion matrix
from sklearn.linear_model import LogisticRegression, _logistic
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import metaestimators

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

import pickle

# #keras
# import tensorflow as tf
# from tensorflow import keras
# from keras import models
# from keras import layers


#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

#hate speech detection library
from hatespeech import train_classifier, classify

# User Network Analysis
from network import build_network, view_network

#globally define variables
df_neg = pd.DataFrame()
df_hate = pd.DataFrame()
df_offensive = pd.DataFrame()

###########################################
########## User Network Analysis ##########
###########################################
def user_network(df_hate, df_offensive):
    
    num_users = 10
    
    print('Building Network...')
    print(df_hate.head(10))
    print(df_offensive.head(10))
    hate_followers_df = pd.DataFrame()
    hate_user_details_df = pd.DataFrame()
    for index, row in df_hate.iterrows():
        master_followers, master_user_details = build_network(row['username'],'hate', num_users)
        hate_followers_df = pd.concat([hate_followers_df, master_followers], ignore_index=True)
        hate_followers_df.to_csv('Datasets/hate_followers.csv', index=False)
        hate_user_details_df = pd.concat([hate_user_details_df, master_user_details], ignore_index=True)
        hate_user_details_df.to_csv('Datasets/hate_user_details.csv', index=False)
    
    offensive_followers_df = pd.DataFrame()
    offensive_user_details_df = pd.DataFrame()
    for index, row in df_hate.iterrows():
        master_followers, master_user_details = build_network(row['username'],'hate', num_users)
        offensive_followers_df = pd.concat([offensive_followers_df, master_followers], ignore_index=True)
        offensive_followers_df.to_csv('Datasets/offensive_followers.csv', index=False)
        offensive_user_details_df = pd.concat([offensive_user_details_df, master_user_details], ignore_index=True)
        offensive_user_details_df.to_csv('Datasets/offensive_user_details.csv', index=False)

    # view_network("hate")
    
    # view_network("offensive")

### Hate Speech Detection

def hate_Analysis(df_neg):
    cv, clf = train_classifier()
    df_hate = pd.DataFrame()
    df_offensive = pd.DataFrame()
    
    for index, row in df_neg.iterrows():
        result = classify(row['tweet_OG'], cv, clf)
        if result == 'Hate Speech Detected':
            df_hate = pd.concat([df_hate, row.to_frame().T], ignore_index=True) 
        elif result == 'Offensive Language Detected':
            df_offensive = pd.concat([df_offensive, row.to_frame().T], ignore_index=True)
        else:
            continue

    # df_hate.info()
    # df_offensive.info()

    #rename 'tweet' column in df_hate and df_offensive to 'tweet_OG'
    df_hate.rename(columns={'tweet':'tweet_OG'}, inplace=True)
    df_offensive.rename(columns={'tweet':'tweet_OG'}, inplace=True)

    df_hate_Users = most_hate(df_hate)
    df_offensive_USers = most_offensive(df_offensive)
    
    return df_hate_Users, df_offensive_USers
    
# Shows users with the most hateful language used in tweets
# takes in a dataframe of tweets with hateful language
# returns a dataframe of the users with the most hateful language used in tweets
def most_hate(df_hate):
    # sets the number of users to be find
    n = 10
    # gets the most common users
    mostCommonUsers = pd.DataFrame(df_hate['username'].value_counts()[:n].index.tolist(), columns=['username'])
    # gets the number of occurences of the most common users
    occurences = pd.DataFrame(df_hate['username'].value_counts()[:n].tolist(), columns=['occurences'])
    
    # combines the two dataframes
    df = pd.concat([mostCommonUsers, occurences], axis=1)

    #Show users with the most hateful language tweets and exports plot to png file
    plt.barh(df['username'],df['occurences'], height = 0.8, color = 'red')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Users')
    plt.title('Users With Most Hateful Language Used')
    plt.savefig('most_hate.png')
    plt.show()
    
    # returns the dataframe
    return df
    
# Show users with the most offensive language tweets
# takes in a dataframe of tweets with offensive language
# returns a dataframe of the users with the most offensive language used in tweets
def most_offensive(df_offensive):
    # sets the number of users to be find
    n = 10
    # gets the most common users
    mostCommonUsers = pd.DataFrame(df_offensive['username'].value_counts()[:n].index.tolist(), columns=['username'])
    # gets the number of occurences of the most common users
    occurences = pd.DataFrame(df_offensive['username'].value_counts()[:n].tolist(), columns=['occurences'])
    
    # combines the two dataframes
    df = pd.concat([mostCommonUsers, occurences], axis=1)
 
    # Show users with the most offensive language tweets and exports plot to png file
    plt.barh(df['username'], df['occurences'], height= 0.8, color = 'red')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Users')
    plt.title('Users With Most Offensive Language Used')
    plt.savefig('offensive.png')
    plt.show()
    
    # returns the dataframe
    return df

### Sentiment Analysis

def bar_chart(df):
    #create bar chart
    #show labels as string values instead of numbers where 0 = negative, 1 = neutral, 2 = positive
    #create copy of df
    df_copy = df.copy()
    df_copy['Sentiment'].replace({0:'Negative', 1:'Neutral', 2:'Positive'}, inplace=True)
    plt.figure(figsize=(8,6))
    sns.countplot(df['Sentiment'])
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.title('Sentiment Analysis')
    plt.show()

#####################################################################
###### ONLY REQUIRED TO PULL TWEETS AND INITIALIZE CSV DATASET ######
#####################################################################
# #extract tweets from twitter using tweepy and store in csv file
# def create_CSV(fname):
#     #connect to twitter api
#     from dotenv import load_dotenv
#     load_dotenv
#     auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))
#     api = tweepy.API(auth, wait_on_rate_limit=True)
#     # #name of csv file to be created
#     # fname = "dataset.csv"
#     #open csv file
#     with open(fname, 'w', encoding='utf-8') as file:
#         W = csv.writer(file)
#         #write header row to csv file
#         W.writerow(['timestamp', 'tweet_OG', 'username', 'all_hashtags', 'location', 
#                     'followers_count', 'retweet_count', 'favorite_count', 'Sentiment'])
#         #search for tweets with the hashtag or keyword = 'WorldCup'
#         for tweet in tweepy.Cursor(api.search, q='#WorldCup', lang='en', tweet_mode='extended').items(1000):
#             W.writerow([tweet.created_at, tweet.full_text.replace('\n',' ').encode('utf-8'), 
#                         tweet.user.screen_name, [e['text'] for e in tweet._json['entities']['hashtags']], 
#                         tweet.user.location, tweet.user.followers_count, tweet.retweet_count, tweet.favorite_count, ''])
#     #close csv file
#     file.close()


#######################################
########## Utility Functions ##########
#######################################

def clean_data(fname):
    print('Cleaning Data')
    df = pd.read_csv(fname)
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
    df = df.sample(frac=1).reset_index(drop=True)
    df_test = df.copy()

    df['Sentiment'] = df['Sentiment'].map({'Extremely Negative':0,'Negative':0,'Neutral':1,'Positive':2,'Extremely Positive':2})
    df_test['Sentiment'] = df_test['Sentiment'].map({'Extremely Negative':0,'Negative':0,'Neutral':1,'Positive':2,'Extremely Positive':2})

    print(df['Sentiment'].value_counts())

    #crossbalancing dataset using RandomOverSampler to create training x and y
    # used to balance the dataset for training
    ros = RandomOverSampler(random_state=0)
    train_x, train_y = ros.fit_resample(np.array(df['tweet_OG']).reshape(-1, 1), np.array(df['Sentiment']).reshape(-1, 1))
    train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['tweet_OG', 'Sentiment'])

    #split dataset into train, test and validation sets
    X = train_os['tweet_OG']
    y = train_os['Sentiment']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_test = df_test['tweet_OG'].values
    y_test = df_test['Sentiment'].values

    
    #create copies of train, test and validation sets
    y_train_le = y_train.copy()
    y_test_le = y_test.copy()
    y_valid_le = y_valid.copy()

    #encode with one hot encoding
    ohe = preprocessing.OneHotEncoder()
    y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
    y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()
    y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()

    print(f"TRAINING DATA: {X_train.shape[0]}\nVALIDATION DATA: {X_valid.shape[0]}\nTESTING DATA: {X_test.shape[0]}" )
    print()

    #save train, test and validation sets to csv files
    train = pd.DataFrame(list(zip(X_train, y_train_le)), columns = ['tweet_OG', 'Sentiment'])
    train.to_csv('Datasets/train.csv', index=False)
    valid = pd.DataFrame(list(zip(X_valid, y_valid_le)), columns = ['tweet_OG', 'Sentiment'])
    valid.to_csv('Datasets/valid.csv', index=False)
    test = pd.DataFrame(list(zip(X_test, y_test_le)), columns = ['tweet_OG', 'Sentiment'])
    test.to_csv('Datasets/test.csv', index=False)

    #call naive_bayes function
    # naive_bayes(X_train, X_test, y_train_le, y_test_le)
    return df

#function to retrieve x and y values from train, test and validation sets stored in csv files
def get_data():
    df_train = pd.read_csv('Datasets/train.csv')
    df_valid = pd.read_csv('Datasets/valid.csv')
    df_test = pd.read_csv('Datasets/test.csv')

    X_train = df_train['tweet_OG'].values
    y_train_le = df_train['Sentiment'].values

    X_valid = df_valid['tweet_OG'].values
    y_valid = df_valid['Sentiment'].values

    X_test = df_test['tweet_OG'].values
    y_test_le = df_test['Sentiment'].values

    return X_train, y_train_le, X_valid, y_valid, X_test, y_test_le

def confusion_matrix_maker(y_test_le, nb_predictions):
    #confusion matrix
    cm = confusion_matrix(y_test_le, nb_predictions)
    # print(cm)
    #plot confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

def naive_bayes(X_train, X_test, y_train_le, y_test_le):
    clf = CountVectorizer() #converts a collection of text documents to a matrix of token counts
    X_train_cv = clf.fit_transform(X_train) #fit the vectorizer to the training data 
    X_test_cv = clf.transform(X_test) #transform the test data

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv) #fit the transformer to the training data
    X_train_tf = tf_transformer.transform(X_train_cv) #transform the training data
    X_test_tf = tf_transformer.transform(X_test_cv) #transform the test data

    nb_classifier = BernoulliNB() #create a naive bayes classifier
    nb_classifier.fit(X_train_tf, y_train_le) #training model

    nb_predictions = nb_classifier.predict(X_test_tf) #predictions on test data using naive bayes classifier

    print("Naive Bayes Accuracy: ", accuracy_score(y_test_le, nb_predictions)) #accuracy score of naive bayes classifier
    print()
    #classification report
    print(classification_report(y_test_le, nb_predictions, target_names= ['Negative','Neutral','Positive']))
    print()

    # #confusion matrix
    # cm = confusion_matrix(y_test_le, nb_predictions)
    # # print(cm)
    # #plot confusion matrix
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True)
    # plt.xlabel('Predicted')
    # plt.ylabel('Truth')
    # plt.show()

def Model_SVM(X_train, X_test, y_train_le, y_test_le):
    clf = CountVectorizer()
    X_train_cv = clf.fit_transform(X_train)
    X_test_cv = clf.transform(X_test)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)
 
    # Next we can begin building the SVM classifier with rbf kernel
    # we use the same parameters as in the Naive Bayes classifier
    model = svm.SVC(kernel='rbf', C=1, gamma=1)
    model.fit(X_train_tf, y_train_le)
    model.score(X_test_tf, y_test_le)

    # we can now predict the sentiment of the test set
    y_pred = model.predict(X_test_tf)

    # we can now evaluate the performance of the classifier
    print(classification_report(y_test_le, y_pred))
    print("accuracy: ", metrics.accuracy_score(y_test_le, y_pred))

    # #confusion matrix
    # cm = confusion_matrix(y_test_le, y_pred)
    # # print(cm)
    # #plot confusion matrix
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True)
    # plt.xlabel('Predicted')
    # plt.ylabel('Truth')
    # plt.show()



def Model_RandomForest(X_train, X_test, y_train_le, y_test_le):
    clf = CountVectorizer()
    X_train_cv = clf.fit_transform(X_train)
    X_test_cv = clf.transform(X_test)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)

    # Next we can begin building the Random Forest classifier
    # we use the same parameters as in the Naive Bayes classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    model.fit(X_train_tf, y_train_le)
    model.score(X_test_tf, y_test_le)

    # we can now predict the sentiment of the test set
    y_pred = model.predict(X_test_tf)

    # we can now evaluate the performance of the classifier
    print(classification_report(y_test_le, y_pred))
    print("accuracy: ", metrics.accuracy_score(y_test_le, y_pred))

    #confusion matrix
    cm = confusion_matrix(y_test_le, y_pred)
    #call plot_confusion_matrix function
    plot_confusion_matrix(cm, classes=['Negative', 'Neutral', 'Positive'], normalize=True, title='Normalized confusion matrix')

    #plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_le, y_pred, pos_label=2)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    #plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test_le, y_pred, pos_label=2)
    plt.plot(recall, precision, color='orange', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    #show wordcloud
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(str(X_test))
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()



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
    print("Cleaning tweets...")
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

#create wordcloud
def wordcloud(df):
    #create wordcloud
    all_words = ' '.join([text for text in df['tweet_OG']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

#plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    # create_CSV()
    # df = clean_data('tweets.csv')
    X_train, y_train_le, X_valid, y_valid, X_test, y_test_le = get_data()
    # run naive bayes classifier
    # start = time.time()
    naive_bayes(X_train,X_test, y_train_le, y_test_le)
    # print("Naive Bayes took: ", time.time()-start)
    #df = pd.read_csv('Datasets/tweets.csv')
    # most_hate, most_offensive = hate_Analysis(df)
    # user_network(most_hate, most_offensive)
    #view_network('hate')
    # start = time.time()
    Model_SVM(X_train, X_test, y_train_le, y_test_le)
    # print("SVM took: ", time.time()-start)
    # df = pd.read_csv('Datasets/tweets.csv')
    # most_hate, most_offensive = hate_Analysis(df)
    # user_network(most_hate, most_offensive)
    # Model_RandomForest(X_train, X_test, y_train_le, y_test_le)
    # wordcloud(df)
    # bar_chart(df)
    # df.info()
    # print(df.head())


if __name__ == "__main__":
    main()





    








