import os
import glob
import math
import random
import pickle
from collections import defaultdict

import hickle
from tqdm import tqdm, trange
import numpy as np

PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'


def get_review_data(users, items, ratings, review_data):
  new_users = []
  new_items = []
  new_ratings = []
  new_photos = []
  new_reviews = []

  for user, item, rating in zip(users, items, ratings):
    for photo_id, reviews in review_data[(user, item)]:
      new_users.append(user)
      new_items.append(item)
      new_ratings.append(rating)
      new_photos.append(photo_id)
      new_reviews.append(reviews)

  return new_users, new_items, new_ratings, new_photos, new_reviews


def batch_review_normalize(reviews, max_length=None):
  batch_size = len(reviews)

  if max_length:
    reviews = [review[:max_length] for review in reviews]
  else:
    max_length = max([len(review) for review in reviews])

  norm_reviews = np.zeros(shape=[batch_size, max_length], dtype=np.int32)  # == PAD
  for i, review in enumerate(reviews):
    for j, word in enumerate(review):
      norm_reviews[i, j] = word

  return norm_reviews


class DataReader:

  def __init__(self, data_dir, training_shuffle=True):
    self.data_dir = data_dir
    self.is_shuffle = training_shuffle
    self.total_users = len(self._read_ids(os.path.join(data_dir, 'users.txt')))
    self.total_items = len(self._read_ids(os.path.join(data_dir, 'items.txt')))
    print('Total users: {}, total items: {}'.format(self.total_users, self.total_items))

    train_data = self._read_data(os.path.join(data_dir, 'train.pkl'))
    test_data = self._read_data(os.path.join(data_dir, 'test.pkl'))
    self.train_rating, self.train_review = self._prepare_data(train_data, training=True)
    self.test_rating, self.test_review = self._prepare_data(test_data)

    self.global_rating = np.asarray(self.train_rating)[:, 2].mean()
    print('Global rating: {:.2f}'.format(self.global_rating))

    self.load_images()

  def load_images(self):
    self.train_id2idx = self._read_img_id2idx(os.path.join(self.data_dir, 'train.id_to_idx.pkl'))
    self.train_img_features = self._read_img_feature(os.path.join(self.data_dir, 'img_feats/train'),
                                                     len(self.train_id2idx.keys()))

    self.test_id2idx = self._read_img_id2idx(os.path.join(self.data_dir, 'test.id_to_idx.pkl'))
    self.test_img_features = self._read_img_feature(os.path.join(self.data_dir, 'img_feats/test'),
                                                    len(self.test_id2idx.keys()))

  def read_train_set(self, batch_size, rating_only=False):
    if self.is_shuffle:
      random.shuffle(self.train_rating)
    if rating_only:
      return self.batch_iterator(self.train_rating, batch_size, True, desc='Training')
    return self.batch_iterator(self.train_review, batch_size, desc='Training')

  def read_test_set(self, batch_size, rating_only=False):
    if rating_only:
      return self.batch_iterator(self.test_rating, batch_size, True, desc='Testing')
    return self.batch_iterator(self.test_review, batch_size, desc='Testing')

  def batch_iterator(self, data, batch_size, rating_only=False, desc=None):
    num_batches = int(math.ceil(len(data) / batch_size))
    self.iter = trange(num_batches, desc=desc)
    for cur_batch in self.iter:
      begin = batch_size * cur_batch
      end = batch_size * cur_batch + batch_size
      if end > len(data):
        end = len(data)

      batch_users = []
      batch_items = []
      batch_ratings = []
      batch_photos = []
      batch_reviews = []

      for exp in data[begin:end]:
        batch_users.append(exp[0])
        batch_items.append(exp[1])
        batch_ratings.append(exp[2])

        if not rating_only:
          batch_photos.append(exp[3])
          batch_reviews.append(exp[4])

      if rating_only:
        yield batch_users, batch_items, batch_ratings
      else:
        yield batch_users, batch_items, batch_ratings, batch_photos, batch_reviews

  @staticmethod
  def _read_ids(file_path):
    print('Reading data: %s' % file_path)
    data = [0]
    with open(file_path, 'r') as f:
      for line in f:
        data.append(int(line.split()[1]))
    return set(data)

  @staticmethod
  def _read_img_feature(feat_dir, num_imgs):
    print('Reading image features: %s' % feat_dir)
    all_feats = np.ndarray([num_imgs, 196, 512], dtype=np.float32)
    for file_path in tqdm(glob.glob('{}/*.hkl'.format(feat_dir))):
      start = int(file_path.split('/')[-1].split('_')[0])
      end = int(file_path.split('/')[-1].split('_')[1].split('.')[0])
      all_feats[start:end, :] = hickle.load(file_path)
    return all_feats

  @staticmethod
  def _read_img_id2idx(file_path):
    print('Reading image id_to_idx: %s' % file_path)
    with open(file_path, 'rb') as f:
      return pickle.load(f)

  @staticmethod
  def _read_data(file_path):
    print('Reading data: %s' % file_path)
    data = []
    with open(file_path, 'rb') as f:
      try:
        while True:
          exp = pickle.load(f)
          data.append(exp)
      except EOFError:
        pass
    return data

  @staticmethod
  def _prepare_data(data, training=False):
    rating_data = []
    review_data = defaultdict(list)

    for exp in data:
      user = int(exp['User'])
      item = int(exp['Item'])
      rating = exp['Rating']
      rating_data.append((user, item, rating))

      for photo_id, photo_reviews in exp['Reviews'].items():
        if training:
          for photo_review in photo_reviews:
            review_data[(user, item)].append((photo_id, photo_review))
        else:
          review_data[(user, item)].append((photo_id, photo_reviews))

    return rating_data, review_data
