import nltk
from nltk.corpus import treebank
from collections import defaultdict, Counter

nltk.download('treebank')

data = treebank.tagged_sents()

trans = defaultdict(Counter)
emit = defaultdict(Counter)

for sent in data:
    prev = "<s>"
    for word, tag in sent:
        trans[prev][tag] += 1
        emit[tag][word] += 1
        prev = tag

def tag_sentence(sentence):
    result = []
    for word in sentence:
        best_tag = max(emit, key=lambda t: emit[t][word])
        result.append((word, best_tag))
    return result

sentence = ["The", "market", "is", "rising"]
print(tag_sentence(sentence))
