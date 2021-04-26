import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

path_to_file = 'grimms.txt'
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')
# length of text is the number of characters in it
# print(f'Length of text: {len(text)} characters')
print(text[:250], "\n")


vocab = sorted(set(text))
# print(vocab)
print(vocab)

ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab))

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)

def text_from_ids(ids):
	return tf.strings.reduce_join(chars_from_ids(ids), axis = -1)
# def to_ids(chars):
# 	return [ord(i) for i in chars]

# def to_chars(ids):
# 	return [chr(i) for i in ids]

# print(to_ids(vocab), to_chars(to_ids(vocab)))
window = 100
char_per_epoch = len(text)//(window+1)
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
	print(chars_from_ids(ids).numpy().decode('utf-8'))

batches = ids_dataset.batch(window, drop_remainder = True)
for ids in batches.take(5):
	print(text_from_ids(ids).numpy())





