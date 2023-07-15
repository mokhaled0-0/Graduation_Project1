import string
import re
import pandas as pd
from tqdm import tqdm  # adding progress bars to show the processing
import preprocessor as p
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))',
    ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p',
    ':-b', ':b', '>:)', '>;)', '>:-)', '<3'
])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)


# combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def clean_tweets(tweet):

    tweet = tweet.lower()  # lower the words to be in the same format

    # after tweepy preprocessing the colon symbol left remain after
    tweet = re.sub(r':', '', tweet)

    tweet = re.sub(r',ÄI', '', tweet)  # removing mentions

    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

    tweet = emoji_pattern.sub(r'', tweet)  # remove emojis from tweet

    tweet = re.sub('[0-9]+', '', tweet)  # remove numbers
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(f'[{string.punctuation}]', '', tweet)  # remove punctuation

    stop_words = set(stopwords.words('english'))  # get the stop words

    # extract the tokens from string of characters
    word_tokens = word_tokenize(tweet)

    difficult_to_detect = ["'re", "'s", "'m", "'ve", "n't", "...", "``", "'", "im",
                           "ca", "itv", "-", "a.", "dont", "us", "could", "can", "'d", "__", 'aaron', 'ab',
                           'zurab', 'zwart', 'zyl', "ll", "u", '__', '___', '____']  # words difficult to detect by the preprocessing modules

    filtered_tweet = []

    # looping through conditions to filter the words
    for w in word_tokens:
        # check tokens against stop words, emoticons and words difficult to detect
        if w not in stop_words and w not in emoticons and w not in difficult_to_detect:
            if len(w) > 3:  # remove the word if it less than 4 character
                w = lemmatizer.lemmatize(w)  # Apply lemmatization on the word
                w = stemmer.stem(w)  # Apply stemming on the word
                # Append the pure word to the list after cleaning
                filtered_tweet.append(w)

    return ' '.join(filtered_tweet)  # Reconstruct the tweet after cleaning
