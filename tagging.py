import nltk
from nltk.corpus import state_union

import spacy
import pandas as pd
import nltk
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('state_union')

#loading the spacy pretrained model on English news
# nlp = spacy.load('en_core_web_md')
# df1 = pd.read_csv('zomato.csv')

# col_list = ['name','reviews_list']
# df = df1[col_list]
# df.to_csv('zomatoclean.csv')
# df = pd.read_csv('zomatoclean.csv')
# df = df[:100]
# name = df['name']
# review = df['reviews_list']

analyser = SentimentIntensityAnalyzer()
def sentiment(s):
    score = analyser.polarity_scores(s)
    listscore = list(score.values())
    return listscore

# for i in range(0,review.size):
#     s = str(review.loc[i])
#     f = sentiment(s)
#     df.loc[i, 'neg_score'] = f[0]*100
#     #df.loc[i, 'neut_score'] = f[1]*100
#     df.loc[i, 'pos_score'] = f[2]*100
#     #df.loc[i, 'comp_score'] = f[3]*100
    
# df.to_csv('zomatoscore.csv')

def POS_Tagging():
    try:
        df = pd.read_csv('zomatoscore.csv')
        neg = df['neg_score']
        pos = df['pos_score']
        reviews = df['reviews_list']
        for i in range(0,reviews.size):
            words = nltk.word_tokenize(str(reviews.loc[i]))
            tagged = nltk.pos_tag(words)
            key_entities = list(filter(lambda word: word[1]=='JJ' or word[1]=='JJR' or word[1]=='JJS' or word[1]=='VB' or word[1]=='VBD' or word[1]=='VBG' or word[1]=='VBN' or word[1]=='VBP' or word[1]=='VBZ', tagged))
            word1 = ''
            word2 = ''
            score1=0.0
            score2=0.0
            if(neg.loc[i]<pos.loc[i]):
                # s = str(review.loc[i])
                # wlist = s.split()
                for entity in key_entities:
                    f = sentiment(entity[0])
                    if(score1<=f[2]*100):
                        score1=f[2]*100
                        word1 = entity[0]
                        
                    if(score2<=f[2]*100 and score2!=score1):
                        score2=f[2]*100
                        word2 = entity[0]
                #df.loc[i,'Max_pos_score'] = maxscore
                
            elif(neg.loc[i]>=pos.loc[i]):
                for entity in key_entities:
                    f = sentiment(entity[0])
                    if(score1<=f[0]*100):
                        score1=f[0]*100
                        word1 = entity[0]

                    if(score2<=f[0]*100 and score2!=score1):
                        score2=f[0]*100
                        word2 = entity[0]
                #df.loc[i,'Max_neg_score'] = minscore

            df.loc[i,'Max_contributing_word'] = word1
            df.loc[i,'Secondmax_contributing_word'] = word2

        df.to_csv('zomatoword.csv')
        df = pd.read_csv('zomatoword.csv')
        print(df)  

    except exception as e:
        print(str(e))                

# sentences = state_union.raw("2005-GWBUsh.txt").split('\n')

POS_Tagging()