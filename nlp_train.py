import tensorflow as tf
print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print()
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras
from keras.layers.experimental import preprocessing
from keras.layers import Dense, GRU, LSTM, RNN, Embedding, Dropout
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
import os
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

def split_into_batches(x,y, batch_size, seq_length):
	remainder = len(x)%batch_size
	x, y = x[:-remainder], y[:-remainder]
	# new_x, new_y = [], []
	# for i in range(0, len(x)-batch_size, batch_size):
	x = np.reshape(x, (len(x)//batch_size, batch_size, seq_length))
	y = np.reshape(y, (len(y)//batch_size, batch_size, y.shape[1]))
	return x, y

# x, y = split_into_batches(x,y,64,seq_length)

print(x.shape, y.shape)
# exit()

def get_model(in_shape, out_shape):
	model = Sequential()
	model.add(GRU(128, input_shape=in_shape))
	# model.add(Dropout(0.2))
	# model.add(GRU(256))
	# model.add(Dropout(0.2))
	model.add(Dense(out_shape, activation='softmax'))
	return model

model = get_model((x.shape[1], x.shape[2]), y.shape[1])
model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.summary()
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint_dir = './my_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_prefix,
	save_weights_only=True)

history = model.fit(x[100:], y[100:], epochs=1, callbacks=[checkpoint_callback])

model.save('my_model')

loaded = keras.models.load_model('my_model')
loaded.summary()