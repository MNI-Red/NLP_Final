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
from keras import callbacks, Model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
import os
from random import sample
print()

task = input('wut: ')
path_to_text_file = 'grimms.txt'
text = open(path_to_text_file, 'r', encoding='utf-8').read()
vocab = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(vocab))
int_to_char = dict((i,c) for i, c in enumerate(vocab))
n_chars, vocab_size = len(text), len(vocab)
print("Chars in text:", n_chars, "\nChars in vocab:", vocab_size, "---", vocab)

if task == 't':
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
	
	# print(x.shape, y.shape)
	zipped = list(zip(x,y))
	sampled_data = sample(zipped, len(x)//10)
	unzipped = list(zip(*sampled_data))
	x, y = np.array(unzipped[0]), np.array(unzipped[1])
	# x = np.reshape(x, (x.shape[0], x.shape[1]))
	print(x.shape, y.shape)

	def split_into_batches(x,y, batch_size, seq_length):
		remainder = len(x)%batch_size
		x, y = x[:-remainder], y[:-remainder]
		# new_x, new_y = [], []
		# for i in range(0, len(x)-batch_size, batch_size):
		x = np.reshape(x, (len(x)//batch_size, batch_size, seq_length))
		y = np.reshape(y, (len(y)//batch_size, batch_size, y.shape[1]))
		return x, y

	def get_model(in_shape):
		model = Sequential()
		model.add(GRU(512, input_shape=in_shape))
		model.add(Dense(len(vocab), activation='softmax'))
		return model
		# inp = tf.keras.Input(shape=in_shape)
		# embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		# gru, state = GRU(512, return_sequences = True, return_state= True)(embedding)
		# dense = Dense(len(vocab), activation='softmax')(gru)
		# return Model(inputs=inp, outputs=dense)
		


	class MyModel(tf.keras.Model):
		def __init__(self, vocab_size, in_shape):
			super().__init__(self)
			self.gru = tf.keras.layers.GRU(512,
										   return_sequences=True,
										   return_state=True)
			self.dense = tf.keras.layers.Dense(vocab_size)

		def call(self, inputs, states=None, return_state=False, training=False):
			x = inputs
			if states is None:
			  states = self.gru.get_initial_state(x)
			x, states = self.gru(x, initial_state=states, training=training)
			x = self.dense(x, training=training)

			if return_state:
			  return x, states
			else:
			  return x
	try:
		model = keras.models.load_model('my_model')
	except FileNotFoundError:
		model = get_model((x.shape[1], x.shape[2]))
		
	# model = MyModel(len(vocab), x.shape)
	# print(x.shape, y.shape)
		
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# model.summary()
	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint_dir = './my_training_checkpoints'
	# Name of the checkpoint files
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_prefix,
		save_weights_only=True)

	history = model.fit(x, y, epochs=10, callbacks=[checkpoint_callback])

	model.save('my_model')
else:
	loaded = keras.models.load_model('my_model')
	print(loaded.summary())
	
	gru = loaded.get_layer('gru')
	dense = loaded.get_layer('dense')

	seed = "The King"
	x = tf.convert_to_tensor([char_to_int[char] for char in seed])
	x = np.reshape(x, (1,len(x),1))
	print(x)
	results = seed
	y = loaded.predict_step(x)
	y = vocab[np.argmax(y)]
	results += vocab[np.argmax(y)]
	print(vocab, y, np.argmax(y))
	# states = []
	state = None
	for i in range(500):
		x = tf.convert_to_tensor([char_to_int[char] for char in results])
		x = np.reshape(x, (1,len(x),1))
		# print(x, state)
		y, state_h = gru(x, initial_state = state)
		y = dense(y)
		print(y, )
		y = loaded.predict_step(x)
		# state = loaded.get_layer('gru').states
		# states.append(state)
		
		results += vocab[np.argmax(y)]
	# print(states)

	from collections import Counter
	counts = Counter(results)
	print(results, '\n', counts, '\n')
	with open("my_output.txt", 'a') as f:
		f.write(results)
		f.write('\n')
		f.write(str(counts))
		f.write('\n')
