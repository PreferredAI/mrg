import os
import _pickle as pkl

import numpy as np
import tensorflow as tf

from reader import PAD_WORD, START_WORD, END_WORD

FLAGS = tf.flags.FLAGS


def log_info(log_file, msg):
  print(msg)
  log_file.write('{}\n'.format(msg))


def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


def count_parameters(trained_vars):
  total_parameters = 0
  print('=' * 100)
  for variable in trained_vars:
    variable_parameters = 1
    for dim in variable.get_shape():
      variable_parameters *= dim.value
    print('{:70} {:20} params'.format(variable.name, variable_parameters))
    print('-' * 100)
    total_parameters += variable_parameters
  print('=' * 100)
  print("Total trainable parameters: %d" % total_parameters)
  print('=' * 100)


def load_glove(vocab_size, embedding_size):
  print('Loading pre-trained word embeddings')
  embedding_weights = {}
  f = open('glove.6B.{}d.txt'.format(embedding_size), encoding='utf-8')
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_weights[word] = coefs
  f.close()
  print('Total {} word vectors in Glove 6B {}d.'.format(len(embedding_weights), embedding_size))

  embedding_matrix = np.random.normal(0, 0.01, (vocab_size, embedding_size))

  vocab = pkl.load(open(os.path.join(FLAGS.data_dir, 'vocab.pkl'), 'rb'))
  oov_count = 0
  for word, i in vocab.items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
    else:
      oov_count += 1
  print('Number of OOV words: %d' % oov_count)

  return embedding_matrix


def load_vocabulary(data_dir):
  vocab_file = os.path.join(data_dir, 'vocab.pkl')
  if not os.path.exists(vocab_file):
    raise FileNotFoundError('Vocabulary not found: %s' % vocab_file)

  print('Reading vocabulary: %s' % vocab_file)
  try:
    with open(vocab_file, 'rb') as f:
      return {idx: word for word, idx in pkl.load(f).items()}
  except IOError:
    pass


def decode_reviews(reviews, vocab):
  if reviews.ndim == 1:
    T = reviews.shape[0]
    N = 1
  else:
    N, T = reviews.shape

  decoded = []
  for i in range(N):
    words = []
    for t in range(T):
      if reviews.ndim == 1:
        word = vocab[reviews[t]]
      else:
        word = vocab[reviews[i, t]]
      if word == END_WORD:
        break
      if word != START_WORD and word != PAD_WORD:
        words.append(word)
    decoded.append(words)
  return decoded
