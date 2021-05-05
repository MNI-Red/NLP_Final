import tensorflow as tf
print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print()
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config)

import string
from tensorflow import keras
from keras.layers.experimental import preprocessing
from keras.layers import Dense, GRU, LSTM, RNN, Embedding, Dropout, BatchNormalization
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import np_utils

import numpy as np
import os
import time
print()
path_to_text_file = 'grimms.txt'
text = open(path_to_text_file, 'r', encoding='utf-8').read()
vocab = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(vocab))
n_chars, vocab_size = len(text), len(vocab)
print("Chars in text:", n_chars, "\nChars in vocab:", vocab_size, "---", vocab)

def split_sequences(seq_length, text, vocab, char_to_int):
	x, y = [], [], 
	for i in range(0, len(text) - seq_length, 1):
		sentence = text[i : i + seq_length]
		next_char = text[i + seq_length]
		x.append([char_to_int[char] for char in sentence])
		y.append(char_to_int[next_char])
	return x,y

seq_length = 100
x, y = split_sequences(seq_length, text, vocab, char_to_int)
num_train_examples = len(x)
print("Num training examples:", num_train_examples)
x = np.reshape(x, (num_train_examples, seq_length, 1))
x = x / float(vocab_size)
y = np_utils.to_categorical(y)

def get_model(in_shape, out_shape):
	model = Sequential()
	model.add(GRU(256, return_sequences=True, input_shape=in_shape))
	model.add(Dropout(0.2))
	model.add(GRU(256))
	model.add(Dropout(0.2))
	model.add(Dense(out_shape, activation='softmax'))
	return model

model = get_model((x.shape[1],x.shape[2]), y.shape[1])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(x, y, epochs=20, batch_size=128, callbacks=callbacks_list)