import re
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer
import itertools
import numpy as np
import nltk
import joblib
import preprocessor as p
from .data_collection_and_preprocessing import clean_tweets
import warnings
import concurrent.futures
from .Sentiment import preprocess
import pandas as pd
import pickle
import joblib


warnings.filterwarnings('ignore')


class sentiment_analysis_code():

    def predicts(self, text):
        file_path = 'C:/Users/ALRYADA/Desktop/sent model/Sentiment-Lm.pkl'
        file_path1 = 'C:/Users/ALRYADA/Desktop/sent model/vectoriser-ngran.pkl'
        model = joblib.load(file_path)
        vectoriser = joblib.load(file_path1)
        # Predict the sentiment
        textdata = vectoriser.transform(preprocess(text))
        sentiment = model.predict(textdata)
        prop = model.predict_proba(textdata)
        probabilty_negative = prop[:, 0]
        probabilty_positive = prop[:, 1]
        # Make a list of text with sentiment.
        data = []
        for text, pred, negative, positive in zip(text, sentiment, probabilty_negative, probabilty_positive):
            data.append((text, pred, negative, positive))

        # Convert the list into a Pandas DataFrame.
        df = pd.DataFrame(
            data, columns=['text', 'sentiment', 'negative', 'positive'])
        df = df.replace([0, 1], ["negative", "positive"])
        print(df.iloc[0]['sentiment'], df.iloc[0]['text'],
              probabilty_negative[0], probabilty_positive[0])
        return df.iloc[0]['sentiment'], df.iloc[0]['text'], probabilty_negative[0], probabilty_positive[0]


file_path = 'C:/Users/ALRYADA/Desktop/sent model/Multinomial_Naive_Bayes.pkl'
file_path1 = 'C:/Users/ALRYADA/Desktop/sent model/SGD.pkl'

mnb = joblib.load(file_path)
sgd = joblib.load(file_path1)


class depression_analysis_code():

    def get_tweet_depression(self, tweet, model1=mnb, model2=sgd):
        # cleaning of tweet
        tweet = clean_tweets(tweet)
        label = ['Not Depressed', 'Depressed']
        probabilities1 = np.array(model1.predict_proba([tweet]))
        probabilities2 = np.array(model2.predict_proba([tweet]))

        # Calculate the average of the two probabilities
        avg_pop = (probabilities1 + probabilities2)/2

        # Determine the final prediction label based on the average probability
        if avg_pop[0][0] > avg_pop[0][1]:
            predic = label[0]

        else:
            predic = label[1]
        print(avg_pop[0][1], avg_pop[0][0], predic)
        # Return the final prediction probability and the class
        return avg_pop[0][1], avg_pop[0][0], predic
