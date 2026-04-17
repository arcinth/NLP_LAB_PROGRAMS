import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

nltk.download('treebank')

data = treebank.tagged_sents()
train = data[:3000]
test = data[3000:]

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train)
hmm_acc = hmm_tagger.evaluate(test)
print("HMM Accuracy:", hmm_acc)

def features(word):
    return {
        'word': word,
        'suffix': word[-2:]
    }

X, y = [], []
for sent in train:
    for word, tag in sent:
        X.append(features(word))
        y.append(tag)

model = Pipeline([
    ('vec', DictVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

model.fit(X, y)

correct = 0
total = 0

for sent in test:
    words = [w for w, t in sent]
    tags = [t for w, t in sent]
    pred = model.predict([features(w) for w in words])

    for p, t in zip(pred, tags):
        if p == t:
            correct += 1
        total += 1

log_acc = correct / total
print("Log-Linear Accuracy:", log_acc)
