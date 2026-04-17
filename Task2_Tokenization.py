from transformers import AutoTokenizer
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

text = "Natural Language Processing is an exciting field of Artificial Intelligence."

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens_transformers = tokenizer.tokenize(text)
print("Transformers Tokens:")
print(tokens_transformers)

tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens_nltk = [word for word in tokens if word.lower() not in stop_words]
print("\nNLTK Tokens (Stopwords Removed):")
print(filtered_tokens_nltk)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
filtered_tokens_spacy = [token.text for token in doc if not token.is_stop]
print("\nspaCy Tokens (Stopwords Removed):")
print(filtered_tokens_spacy)
