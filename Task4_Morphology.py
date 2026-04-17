import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')

text = "The boys are playing games and running faster than before."

tokens = word_tokenize(text.lower())

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

print("NLTK Stemming Output:")
for word, stem in zip(tokens, stemmed_words):
    print(f"{word} -> {stem}")

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print("\nspaCy Morphological Analysis:")
for token in doc:
    print(f"Word: {token.text}")
    print(f" Lemma: {token.lemma_}")
    print(f" POS: {token.pos_}")
    print(f" Morphology: {token.morph}")
    print()
