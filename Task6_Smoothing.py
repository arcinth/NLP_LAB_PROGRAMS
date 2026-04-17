from nltk.util import ngrams
from collections import Counter

text = "natural language processing is fun and language processing is powerful"
tokens = text.lower().split()

n = 2
ngrams_list = list(ngrams(tokens, n))

ngram_counts = Counter(ngrams_list)
unigram_counts = Counter(tokens)

V = len(set(tokens))

def laplace_prob(w1, w2):
    return (ngram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)

print("Bigram Probabilities with Laplace Smoothing:\n")
for (w1, w2) in ngram_counts:
    prob = laplace_prob(w1, w2)
    print(f"P({w2} | {w1}) = {prob:.4f}")
