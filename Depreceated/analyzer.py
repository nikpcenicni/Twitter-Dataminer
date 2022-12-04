import string
from collections import Counter

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

import re, emoji



def clean_tweets():
    text = open('tweet_list.txt', encoding='utf-8').read()
    
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    text = re.sub(r"https\S+|www\S+http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'ð','',text)
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)

    # text = re.sub(emoji.get_emoji_regexp(), r"", text) #remove emoji

    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    return cleaned_text
    
def preprocess_text():
    cleaned_text = clean_tweets()
    # Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text, "english")
    final_words = stop_words(tokenized_words)
    lemma_words = lemmatize_words(final_words)
    emotion_words = emotion_analysis(lemma_words)
    return emotion_words

def stop_words(tokenized_words):
    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)
    return final_words

def lemmatize_words(final_words):
    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)
    return lemma_words

def emotion_analysis(lemma_words):
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in lemma_words:
                emotion_list.append(emotion)

    # print(emotion_list)
    w = Counter(emotion_list)
    # print(w)
    return w

def sentiment_analysis(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        return("Negative Sentiment")
    elif score['neg'] < score['pos']:
        return("Positive Sentiment")
    else:
        return("Neutral Sentiment")



def graph_emotions(w):
    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values())
    fig.autofmt_xdate()
    plt.savefig('graph.png')
    plt.show()
    
def analyse_tweets():
    cleaned_text = clean_tweets()
    return sentiment_analysis(cleaned_text)
    # emotion_list = preprocess_text()
    # graph_emotions(emotion_list)
    # return
