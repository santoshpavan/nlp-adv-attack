import nltk
from nltk.corpus import state_union
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('state_union')

sentences = state_union.raw("2005-GWBUsh.txt").split('\n')

def POS_Tagging():
    try:
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(words)
            adjectives = list(filter(lambda word: word[1]=='JJ' or word[1]=='JJR' or word[1]=='JJS', tagged))
            print(adjectives)

    except exception as e:
        print(str(e))

POS_Tagging()