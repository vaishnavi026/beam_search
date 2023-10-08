import nltk
nltk.download('punkt')

from nltk import FreqDist
from math import log


# Preprocess and tokenize the text
def preprocess(text):
    return nltk.word_tokenize(text.lower())


def get_prob_dist(texts):
    words = []
    for text in texts:
        words.extend(preprocess(text))

    freq_dist = FreqDist(words)
    prob_dist = {word: freq / len(words) for word, freq in freq_dist.items()}
    return prob_dist


def kl_divergence(P, Q, epsilon=1e-10):
    """
    Compute KL divergence between two distributions, P and Q.
    Add epsilon to handle 0 probabilities.
    """
    return sum(P[word] * log((P[word] + epsilon) / (Q.get(word, 0) + epsilon)) for word in P)


texts1 = ["Text 1 from data set 1 Text 2 from data set 1"]
texts2 = ["Totally unrelated text Another unrelated text"]

