from django.shortcuts import render, redirect, HttpResponse
from .forms import Sentiment_Typed_Tweet_analyse_form
from .sentiment_analysis_code import sentiment_analysis_code, depression_analysis_code
from .forms import Sentiment_Imported_Tweet_analyse_form
from .tweepy_sentiment import Import_tweet_sentiment
from plotly.offline import plot
from plotly.graph_objs import Scatter
import plotly.express as px
import pandas as pd
from textblob import TextBlob


def tweet(request):

    return render(
        request=request,
        template_name='home/tweet.html',

    )


def sentiment_analysis(request):
    return render(request, 'home/sentiment.html')


def sentiment_analysis_type(request):
    if request.method == 'POST':
        form = Sentiment_Typed_Tweet_analyse_form(request.POST)
        analyse = sentiment_analysis_code()
        if form.is_valid():
            tweet = form.cleaned_data['sentiment_typed_tweet']
            sentiment = analyse.predicts([tweet])
            if abs(sentiment[3] - .63) <= .001:
                sentiment = (sentiment[0], sentiment[1], .5, .5)
            lab = sentiment[0]
            args = {'tweet': tweet, 'sentiment': lab}
            Labels = ['negative', 'positive']
            data = [sentiment[2], sentiment[3]]
            fig = px.pie(names=Labels, values=data, color=Labels,
                         color_discrete_map={'positive': '#39e75f',
                                             'negative': '#d91515', })
            fig.update_layout(
                {'paper_bgcolor': 'rgba(0, 0, 0 ,0)'})
            fig.update_layout(width=int(620))
            p = fig.to_html()
            args["p"] = p
            return render(request, 'home/sentiment_type_result.html', args)

    else:
        form = Sentiment_Typed_Tweet_analyse_form()
        return render(request, 'home/sentiment_type.html')


def Depression_analysis_type(request):
    if request.method == 'POST':
        form = Sentiment_Typed_Tweet_analyse_form(request.POST)
        analyse = depression_analysis_code()
        if form.is_valid():
            tweet = form.cleaned_data['sentiment_typed_tweet']
            sentiment = analyse.get_tweet_depression(tweet)
            args = {'tweet': tweet, 'sentiment': sentiment[2]}
            Labels = ['Depressed', 'Not Depressed']
            data = [sentiment[0], sentiment[1]]
            fig = px.pie(names=Labels, values=data, color=Labels,
                         color_discrete_map={'Not Depressed': '#39e75f',
                                             'Depressed': '#d91515', })
            fig.update_layout(
                {'paper_bgcolor': 'rgba(0, 0, 0 ,0)'})
            fig.update_layout(width=int(620))
            p = fig.to_html()
            args["p"] = p
            return render(request, 'home/Depression_res.html', args)

    else:
        form = Sentiment_Typed_Tweet_analyse_form()
        return render(request, 'home/Depression.html')


def sentiment_analysis_import(request):
    if request.method == 'POST':
        form = Sentiment_Imported_Tweet_analyse_form(request.POST)
        tweet_text = Import_tweet_sentiment()
        analyse = sentiment_analysis_code()

        if form.is_valid():
            handle = form.cleaned_data['sentiment_imported_tweet']
            if handle[0] == '#':
                list_of_tweets = tweet_text.get_hashtag(handle, request)
                list_of_tweets_and_sentiments = []
                sent = {}
                sentiment = []

                for i in list_of_tweets:
                    s = analyse.predicts([i])
                    if abs(s[3] - .61) <= .01:
                        s = (s[0], s[1], .5, .5)
                    st = ""
                    if s[3] >= .55:
                        st = 'positive'
                    elif s[3] >= .45:
                        st = 'neutral'
                    else:
                        st = "negative"
                    list_of_tweets_and_sentiments.append((i, st))
                    sent[st] = sent.get(st, 0) + 1
                    sentiment.append(st)
                args = {'tweet': list_of_tweets_and_sentiments,
                        'sentiment': sentiment}
                colorss = sent.keys()
                fig = px.pie(names=colorss, values=sent.values(), color=colorss,
                             color_discrete_map={'positive': '#39e75f',
                                                 'negative': '#d91315',
                                                 'neutral': '#89bdee'})
                fig.update_layout(
                    {'paper_bgcolor': 'rgba(0, 0, 0 ,0)'})
                fig.update_layout(width=int(620))
                p = fig.to_html()
                args = {
                    'list_of_tweets_and_sentiments': list_of_tweets_and_sentiments, 'p': p, 'handle': handle}
                return render(request, 'home/sentiment_import_result_hashtag.html', args)

            list_of_tweets = tweet_text.get_tweets(handle, request)
            list_of_tweets_and_sentiments = []
            if handle[0] != '@':
                handle = str('@'+handle)
            sent = {}
            sentiment = []
            for i in list_of_tweets:
                s = analyse.predicts([i])
                if abs(s[3] - .61) <= .01:
                    s = (s[0], s[1], .5, .5)
                st = ""
                if s[3] >= .55:
                    st = 'positive'
                elif s[3] >= .45:
                    st = 'neutral'
                else:
                    st = "negative"
                list_of_tweets_and_sentiments.append((i, st))
                sent[st] = sent.get(st, 0) + 1
                sentiment.append(st)
            args = {'tweet': list_of_tweets_and_sentiments,
                    'sentiment': sentiment}
            colorss = sent.keys()
            fig = px.pie(names=colorss, values=sent.values(), color=colorss,
                         color_discrete_map={'positive': '#39e75f',
                                             'negative': '#d91515',
                                             'neutral': '#89bdee'})
            fig.update_layout(
                {'paper_bgcolor': 'rgba(0, 0, 0 ,0)'})
            fig.update_layout(width=int(620))
            p = fig.to_html()
            args = {
                'list_of_tweets_and_sentiments': list_of_tweets_and_sentiments, 'p': p, 'handle': handle}
            return render(request, 'home/sentiment_import_result.html', args)

    else:
        form = Sentiment_Imported_Tweet_analyse_form()
        return render(request, 'home/sentiment_import.html')
