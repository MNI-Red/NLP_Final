import tensorflow as tf
import string
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from keras.layers import Dense, GRU, LSTM, RNN, Embedding
from keras import callbacks

import numpy as np
import os
import time

print()
CHUNK_LENGTH = 40
STEP = 3
LEARNING_RATE = 0.0005

path_to_file = 'grimms.txt'
text = open(path_to_file, 'r').read()
corpus_length = len(text)
print(text[:10])

vocab = string.ascii_uppercase + string.ascii_lowercase + string.punctuation + string.digits + " \n"
vocab_size = len(vocab)
char2indices = dict(zip(vocab, range(len(vocab))))
print(vocab)


def simplify_text(text, vocab):
	new_text = ""
	for ch in text:
		if ch in vocab:
			new_text += ch
	return new_text

data = simplify_text(text, char2indices)
print(f"Type of the data is: {type(data)}\n")
print(f"Length of the data is: {len(data)}\n")
print(f"The first couple of sentence of the data are:\n")
print(data[:500])

def encode(char, char2indices):
	return char2indices[char]

def encode_sentence(sent, char2indices):
	return [encode(c, char2indices) for c in sent]

def get_x_y(text, char_indices):
	sentences = []
	next_chars = []
	for i in range(0, len(text) - CHUNK_LENGTH, STEP):
		sentences.append(text[i : i + CHUNK_LENGTH])
		next_chars.append(text[i + CHUNK_LENGTH])

	print("Chunk length:", CHUNK_LENGTH)
	print("Number of chunks:", len(sentences))

	x = []
	y = []
	for i, sentence in enumerate(sentences):
		x.append(encode_sentence(sentence, char_indices))
		y.append(encode(next_chars[i], char_indices))

	return np.array(x, dtype=bool), np.array(y, dtype=bool)

x, y = get_x_y(data, char2indices)
print("Shape of x is", x.shape)
print("Shape of y is ", y.shape)

class MyModel(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, rnn_units):
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(rnn_units,
									   return_sequences=True,
									   return_state=True)
		self.dense = tf.keras.layers.Dense(vocab_size)

	def call(self, inputs, states=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x, training=training)
		if states is None:
			states = self.gru.get_initial_state(x)
		x, states = self.gru(x, initial_state=states, training=training)
		x = self.dense(x, training=training)

		if return_state:
			return x, states
		else:
			return x

embedding_dim = 256
rnn_units = 1024

model = MyModel(
	# Be sure the vocabulary size matches the `StringLookup` layers.
	vocab_size=vocab_size,
	embedding_dim=embedding_dim,
	rnn_units=rnn_units)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

model.summary()