import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

sentences = [["Welcome", "to", "our", "site"], ["This", "product", "is", "great"]]
chunks = [["H", "H", "H", "H"], ["P", "P", "P", "P"]]

words = list(set(w for s in sentences for w in s))
labels = list(set(l for c in chunks for l in c))

word2idx = {w: i + 1 for i, w in enumerate(words)}
label2idx = {l: i for i, l in enumerate(labels)}

X = np.array([[word2idx[w] for w in s] for s in sentences])
y = np.array([[label2idx[l] for l in c] for c in chunks])

model = Sequential()
model.add(Embedding(input_dim=len(words) + 1, output_dim=8))
model.add(LSTM(8, return_sequences=True))
model.add(Dense(len(labels), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=50, verbose=0)

test = np.array([[word2idx["Welcome"], word2idx["product"]]])
pred = model.predict(test)

print("Predicted Chunks:", pred.argmax(axis=-1))
