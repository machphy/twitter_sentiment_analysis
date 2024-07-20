import nltk
from nltk.corpus import twitter_samples
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import pickle
import itertools

nltk.download('twitter_samples')
nltk.download('punkt')

def create_word_features(words):
    score = BigramAssocMeasures.chi_sq
    n = 600
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score, n)
    return dict([(word, True) for word in itertools.chain(words, bigrams)])

positive_temp_tweets = twitter_samples.strings('positive_tweets.json')
negative_temp_tweets = twitter_samples.strings('negative_tweets.json')
positive_tweets = []
negative_tweets = []

for word in positive_temp_tweets:
    str = ""
    j = 0
    while j < len(word):
        if word[j] == '@':
            while j < len(word) and word[j] != ' ' and word[j] != '\n':
                j += 1
        if j < len(word):
            str += word[j]
        j += 1
    positive_tweets.append((create_word_features(word_tokenize(str)), "positive"))

for word in negative_temp_tweets:
    str = ""
    j = 0
    while j < len(word):
        if word[j] == '@':
            while j < len(word) and word[j] != ' ' and word[j] != '\n':
                j += 1
        if j < len(word):
            str += word[j]
        j += 1
    negative_tweets.append((create_word_features(word_tokenize(str)), "negative"))

train_set = negative_tweets[:4000] + positive_tweets[:4000]
test_set = negative_tweets[4000:] + positive_tweets[4000:]


classifier = NaiveBayesClassifier.train(train_set)
classify_buffer = open('twitter_reviews.pickle', 'wb')
pickle.dump(classifier, classify_buffer)
classify_buffer.close()


print("Accuracy is :", nltk.classify.util.accuracy(classifier, test_set) * 100)
