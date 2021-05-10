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
from tensorflow.keras.layers.experimental import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
import time

path_to_file = 'grimms.txt'
# model_name = "my_model"
model_name = "tensorflow_model"

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

model = keras.models.load_model(model_name)

ids_from_chars = preprocessing.StringLookup(
	vocabulary=list(vocab))
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
	vocabulary=ids_from_chars.get_vocabulary(), invert=True)

class OneStep(tf.keras.Model):
		def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
			super().__init__()
			self.temperature = temperature
			self.model = model
			self.chars_from_ids = chars_from_ids
			self.ids_from_chars = ids_from_chars

			# Create a mask to prevent "" or "[UNK]" from being generated.
			skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
			sparse_mask = tf.SparseTensor(
				# Put a -inf at each bad index.
				values=[-float('inf')]*len(skip_ids),
				indices=skip_ids,
				# Match the shape to the vocabulary
				dense_shape=[len(ids_from_chars.get_vocabulary())])
			self.prediction_mask = tf.sparse.to_dense(sparse_mask)

		@tf.function
		@tf.autograph.experimental.do_not_convert
		def generate_one_step(self, inputs, states=None):
			# Convert strings to token IDs.
			input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
			input_ids = self.ids_from_chars(input_chars).to_tensor()

			# Run the model.
			# predicted_logits.shape is [batch, char, next_char_logits]
			predicted_logits, states = self.model(inputs=input_ids, states=states,
												  return_state=True)
			# Only use the last prediction.
			predicted_logits = predicted_logits[:, -1, :]
			predicted_logits = predicted_logits/self.temperature
			# Apply the prediction mask: prevent "" or "[UNK]" from being generated.
			predicted_logits = predicted_logits + self.prediction_mask

			# Sample the output logits to generate token IDs.
			predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
			predicted_ids = tf.squeeze(predicted_ids, axis=-1)

			# Convert from token ids to characters
			predicted_chars = self.chars_from_ids(predicted_ids)

			# Return the characters and model state.
			return predicted_chars, states
	
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
start = time.time()
states = None
next_char = tf.constant(['Once upon a time'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)	
"""
if model_name == "my_model":
	model = keras.models.load_model(model_name)
	char_to_int = dict((c, i) for i, c in enumerate(vocab))
	seed = tf.constant(['Once upon a time'])
	one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

else:
	# Read, then decode for py2 compat.
	one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
	start = time.time()
	states = None
	next_char = tf.constant(['Once upon a time'])
	result = [next_char]

	for n in range(1000):
	  next_char, states = one_step_model.generate_one_step(next_char, states=states)
	  result.append(next_char)

	result = tf.strings.join(result)
	end = time.time()
	print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
	print('\nRun time:', end - start)

	tf.saved_model.save(one_step_model, './models/one_step')
"""