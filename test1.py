import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
import pandas as pd
import re
# to check for valid english words
from nltk.corpus import words
# for the vector creation
import numpy as np
import spacy
import gensim
from gensim.models import Word2Vec
# for a beep after running is over
import winsound

frequency = 750  # 2500 Hz
duration = 500  # 1 sec

"""
TASKS:
1. Data Processing - Tokenize and Stop Words
2. Create vectors for words using Spacy and word2vec
3. Handle new words that aren't included in SpaCy and word2vec
"""

"""
1. Data Processing-
1.1 Split the input of several restaurants into reviews per restaurant - Done
1.2 Filter out the rating values for each review and put them in a matrix of ratings and review for each review.
1.3 Divide them further into single review per index in a list.
1.4 Divide them even more into single sentences - Ignoring!
1.5 Remove those sentences/reviews with the garbage values - need to look into the received output to remove which one.
1.6 Filter those sentences of any stop words.
1.7 Filter them further of any garbage values as observed earlier.
"""
# txt = "keka [ (Before), ( The death shall dance tonight ) we dine.] and kill"
# taking the input from the csv document
zomato = pd.read_csv("zomato.csv")
# taking just the column of the review list
reviews = zomato['reviews_list']
# print(reviews)

# Dividing the multiple reviews in the single cell
# individualReviews = []
reviewBlocks = []
reviews = reviews[0:6]
# print(reviews)
for eachRestaurantReview in reviews:
    """1.1 eachRestaurant has all the reviews of one restaurant each """
    # reviewBlocks contains the each reviewBlock of the restaurant
    # each reviewBlock consists of the rating and the review given in a single entry
    # reviewBlocks = reviewBlocks + re.split('[\(\)\[\]]', eachRestaurantReview)
    # reviewBlocks = reviewBlocks + re.split(r'\(([^(]+)\)', eachRestaurantReview)
    reviewBlocks = reviewBlocks + re.split(r"\(('[^']*'(?:,\s*'[^']*')*)\)", eachRestaurantReview)

# print(reviewBlocks)
# contents in reviewBlocks cleaning - contains empty strings
reviewBlocksFiltered = list(filter(None, reviewBlocks))
# contents in reviewBlocksFiltered cleaning - contains "," but with whitespace
for i in reviewBlocksFiltered:
    # i.replace(" ","") is removing the whitespace
    if i.replace(" ", "") == ",":
        reviewBlocksFiltered.remove(i)
# print(reviewBlocksFiltered)

# Now dividing each element in the list into a list of two - rating and review
filteredBlocksDivided = []
for i in reviewBlocksFiltered:
    # filteredBlocksDivided = filteredBlocksDivided + [re.split('[,]', i)]
    filteredBlocksDivided = filteredBlocksDivided + [i.split(',', 1)]
    # the 1 mentioned in the arguments of split will split only at the first occurrence
print(filteredBlocksDivided)

"""1.2 Filtering out the rating values and putting them in a matrix of ratings and review for each review"""
# Now need to remove the keywords - Rated/RATED because they are unnecessary
for rBlock in filteredBlocksDivided:
    # print(rBlock)
    # print(len(rBlock))
    # There are a few elements that were result of reviews written in brackets
    if len(rBlock) < 2:
        filteredBlocksDivided.remove(rBlock)
    else:
        # Few reviews are listed like, 1) resulting in splitting in the wrong places. Deleting them for now.
        try:
            rBlock[0] = float(rBlock[0].strip("'Rated'").strip())
        except ValueError:
            filteredBlocksDivided.remove(rBlock)
        # the 1 is written since we only want the first instance to be removed
        rBlock[1] = rBlock[1].strip("'RATED', 1").strip()
        rBlock[1] = rBlock[1].replace("\\n", "").strip()

#print(filteredBlocksDivided)
print("Dumb Filter being used...")
# Dumb filter
for i in filteredBlocksDivided:
    if len(i) < 2:
        # print(i)
        filteredBlocksDivided.remove(i)

# Collecting the set of stop words
stopWords = set(stopwords.words('english'))
"""1.6 Filtering the review comments of any stop words. (Ignoring the 1.4)"""
"""1.7 Conducting extra filtering, removing the punctuations that's not covered in 1.6"""

# The punctuationExtra is for the 1.7
punctuationExtra = [".", ",", "''", "``", "(", ")", ":", ";", "{", "}", "[", "]", "!", "~", "`", "?", "'", "/"]

# To check the immediate duplicates. Ignoring the duplicates or else.
reviewTokensOld = []
# has all the review only in tokens
reviewTokensFinal = []

for review in filteredBlocksDivided:
    # Duplicate entries in the reviews. Getting rid of the them. "set" won't work as list is hashable - mutable (here)
    # print(review)
    # comment has the tokenized review[1] and review[0] is the rating
    # adding space after these to get good tokens
    review[1] = review[1].replace(".", " . ")
    review[1] = review[1].replace("'", " ' ")
    review[1] = review[1].replace("/", " ")
    # it's a waste to keep \\n still lying around
    review[1] = review[1].replace("\\n", "")
    reviewTokensNow = word_tokenize(review[1].lower())
    if reviewTokensNow == reviewTokensOld:
        # found a duplicate! Don't think have to remove as under because won't use it anyway
        # filteredBlocksDivided.remove(review)
        continue
    else:
        # if not a duplicate then go on with the process
        # print(reviewTokens)
        # removing the stopwords
        # 1.6 being realised
        reviewTokensFiltered = [i for i in reviewTokensNow if i not in stopWords]
        # print(reviewTokensFiltered)
        # 1.7 being realised
        reviewTokensFurtherFiltered = [i for i in reviewTokensFiltered if i not in punctuationExtra]
        # print(reviewTokensFurtherFiltered)
        # checking if these are corrupted i.e. valid english words - including variations - PROBLEM!!!
        for i in range(0, len(reviewTokensFurtherFiltered)):  # Most of them are variations - PROBLEM!!!
            if reviewTokensFurtherFiltered[i] not in words.words():
                # found one garbage! Don't think have to remove as under because won't use it anyway
                # filteredBlocksDivided.remove(review)
                # print(reviewTokensFurtherFiltered[i])
                i = i+1
            else:
                reviewTokensFinal = reviewTokensFinal + [reviewTokensFurtherFiltered]
    # updating to check later for duplicates
    reviewTokensOldOld = reviewTokensNow

print(reviewTokensFinal)
"""
2. Creating Vectors-
2.1 Create Vectors with Spacy
2.2 Create Vectors with word2vec
"""
reviewVectorSpacyFinal = []
reviewVectorSGFinal = []
nlp = spacy.load('en_core_web_md')

for reviewToken in reviewTokensFinal:
    # Skip Gram is apparently better than CBOW
    reviewVectorSG = gensim.models.Word2Vec(reviewToken, min_count=1, size=100, window=5, sg=1)
    reviewVectorSGFinal = reviewVectorSGFinal + [reviewVectorSG]
    # need to combine the tokens into sentences for word2vec
    reviewTokenCombined = " ".join(reviewToken)
    reviewVectorSpacy = nlp(reviewTokenCombined)
    reviewVectorSpacyFinal = reviewVectorSpacyFinal + [reviewVectorSpacy]

print(reviewVectorSGFinal)
print(reviewVectorSpacyFinal[0].vector)

# Make a sound after done
winsound.Beep(frequency, duration)
