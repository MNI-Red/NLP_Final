import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config)

import string
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense, GRU, LSTM, RNN, Embedding, Dropout, BatchNormalization
from tensorflow.keras import callbacks, Sequential

import numpy as np
import os
import time

print()
CHUNK_LENGTH = 40
LEARNING_RATE = 0.0005
STEP = 3
path_to_text_file = 'grimms.txt'

text = open(path_to_text_file, 'r').read()
# path_to_file = 'grimms.txt'
# text = open(path_to_file, 'r').read()
corpus_length = len(text)
print(text[:10])

vocab = string.ascii_uppercase + string.ascii_lowercase + string.punctuation + string.digits + " \n"
vocab_size = len(vocab)
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))

char2indices = dict(zip(vocab, range(len(vocab))))
print(vocab)
chars = list("yeeticus")
ids = ids_from_chars(chars)
print(ids)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)
print(tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy())

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))


# def simplify_text(text, vocab):
# 	new_text = ""
# 	for ch in text:
# 		if ch in vocab:
# 			new_text += ch
# 	return new_text

# data = simplify_text(text, char2indices)
# print(f"Type of the data is: {type(data)}\n")
# print(f"Length of the data is: {len(data)}\n")
# print(f"The first couple of sentence of the data are:\n")
# print(data[:500])


ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(sequence):
	input_text = sequence[:-1]
	target_text = sequence[1:]
	return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

def get_model(vocab_size, embedding_dim, rnn_units):
	model = Sequential()
	model.add(tf.keras.layers.GRU(rnn_units, return_sequences=True,return_state=True))
	model.add(Dropout(0.3))
	model.add(tf.keras.layers.GRU(rnn_units, return_sequences=True,return_state=True))
	model.add(Dense(vocab_size))
	return model
# x, y = get_x_y(all_ids, ids_from_chars)
# print("Shape of x is", x.shape)
# print("Shape of y is ", y.shape)
# x = x.reshape(170742, 40, 1)
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
rnn_units = 512

# model = MyModel(
# 	# Be sure the vocabulary size matches the `StringLookup` layers.
# 	vocab_size=vocab_size,
# 	embedding_dim=embedding_dim,
# 	rnn_units=rnn_units)

model = get_model(vocab_size, embedding_dim, rnn_units)
# print(x.shape, x[1].shape, y.shape, y[1].shape)
# x=x.reshape(170742,40)

# example_batch_predictions = model(x[1])
# print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# model.summary()

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 20
# history = model.fit(x,y, epochs=EPOCHS, callbacks=[checkpoint_callback])
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
