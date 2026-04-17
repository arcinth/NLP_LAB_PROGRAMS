import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
tags = [["PRP", "VBP", "NN"], ["NN", "VBZ", "JJ"]]

words = list(set(w for s in sentences for w in s))
tags_list = list(set(t for s in tags for t in s))

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags_list)}

X = [[word2idx[w] for w in s] for s in sentences]
y = [[tag2idx[t] for t in s] for s in tags]

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(Embedding(input_dim=len(words) + 1, output_dim=8))
model.add(LSTM(8, return_sequences=True))
model.add(Dense(len(tags_list), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=50, verbose=0)

test = np.array([[word2idx["I"], word2idx["love"], word2idx["NLP"]]])
pred = model.predict(test)

print("Prediction:", pred.argmax(axis=-1))

print("\nInformation Extraction Example:")
for word, tag in zip(["I", "love", "NLP"], ["PRP", "VBP", "NN"]):
    if tag == "NN":
        print("Noun:", word)
