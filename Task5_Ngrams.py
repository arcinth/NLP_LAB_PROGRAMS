import nltk
import random
import spacy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import defaultdict

nltk.download('punkt')

text = """Natural language processing is a field of artificial intelligence.
It enables computers to understand human language."""

tokens = word_tokenize(text.lower())

n = 2
bigrams = list(ngrams(tokens, n))

print("Generated Bigrams:")
print(bigrams)

model = defaultdict(list)
for w1, w2 in bigrams:
    model[w1].append(w2)

def generate_text(seed, num_words):
    current_word = seed.lower()
    output = [current_word]

    for _ in range(num_words):
        next_words = model.get(current_word)
        if not next_words:
            break
        next_word = random.choice(next_words)
        output.append(next_word)
        current_word = next_word

    return " ".join(output)

print("\nGenerated Text:")
print(generate_text("natural", 8))

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print("\nspaCy Tokens:")
print([token.text for token in doc])
