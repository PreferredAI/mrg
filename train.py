from collections import defaultdict

import tensorflow as tf
import numpy as np

from model import Model
from reader import DataReader, get_review_data, batch_review_normalize
from utils import count_parameters, load_vocabulary, decode_reviews, log_info
from bleu import compute_bleu
from rouge import rouge

# Parameters
# ==================================================
tf.flags.DEFINE_string("data_dir", "data",
                       """Path to the data directory""")

tf.flags.DEFINE_float("learning_rate", 3e-4,
                      """Learning rate (default: 3e-4)""")
tf.flags.DEFINE_float("dropout_rate", 0.2,
                      """Probability of dropping neurons (default: 0.2)""")
tf.flags.DEFINE_float("lambda_reg", 1e-4,
                      """Lambda hyper-parameter for regularization (default: 1e-4)""")

tf.flags.DEFINE_integer("num_epochs", 20,
                        """Number of training epochs (default: 20)""")
tf.flags.DEFINE_integer("batch_size", 64,
                        """Batch size of reviews (default: 64)""")
tf.flags.DEFINE_integer("num_factors", 256,
                        """Number of latent factors for users/items (default: 256)""")
tf.flags.DEFINE_integer("word_dim", 200,
                        """Word embedding dimensions (default: 200)""")
tf.flags.DEFINE_integer("lstm_dim", 256,
                        """LSTM hidden state dimensions (default: 256)""")
tf.flags.DEFINE_integer("max_length", 20,
                        """Maximum length of reviews to be generated (default: 20)""")
tf.flags.DEFINE_integer("display_step", 10,
                        """Display info after number of steps (default: 10)""")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

FLAGS = tf.flags.FLAGS


def check_scope_rating(var_name):
  for name in ['user', 'item', 'features', 'rating']:
    if name in var_name:
      return True
  return False


def check_scope_review(var_name):
  for name in ['user', 'item', 'features', 'review']:
    if name in var_name:
      return True
  return False


def train_fn(model):
  global_step = tf.train.create_global_step()

  trainable_vars = tf.trainable_variables()
  count_parameters(trainable_vars)

  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  rating_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars
                             if check_scope_rating(v.name) and 'bias' not in v.name])
  model.rating_loss = model.rating_loss + FLAGS.lambda_reg * rating_l2_loss

  review_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars
                             if check_scope_review(v.name) and 'bias' not in v.name])
  model.review_loss = model.review_loss + FLAGS.lambda_reg * review_l2_loss

  update_rating = optimizer.minimize(model.rating_loss, name='update_rating', global_step=global_step)
  update_review = optimizer.minimize(model.review_loss, name='update_review')

  return update_rating, update_review, global_step


def main(_):
  vocab = load_vocabulary(FLAGS.data_dir)
  data_reader = DataReader(FLAGS.data_dir)

  model = Model(total_users=data_reader.total_users, total_items=data_reader.total_items,
                global_rating=data_reader.global_rating, num_factors=FLAGS.num_factors,
                img_dims=[196, 512], vocab_size=len(vocab), word_dim=FLAGS.word_dim,
                lstm_dim=FLAGS.lstm_dim, max_length=FLAGS.max_length, dropout_rate=FLAGS.dropout_rate)

  update_rating, update_review, global_step = train_fn(model)

  log_file = open('log.txt', 'w')
  test_step = 0

  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, FLAGS.num_epochs + 1):
      log_info(log_file, "\nEpoch: {}/{}".format(epoch, FLAGS.num_epochs))

      count = 0
      sum_rating_loss = 0
      sum_review_loss = 0

      # Training
      for users, items, ratings in data_reader.read_train_set(FLAGS.batch_size, rating_only=True):
        count += 1

        fd = model.feed_dict(users=users, items=items, ratings=ratings, is_training=True)
        _step, _, _rating_loss = sess.run([global_step, update_rating, model.rating_loss], feed_dict=fd)
        sum_rating_loss += _rating_loss

        review_users, review_items, _, photo_ids, reviews = get_review_data(users, items, ratings,
                                                                            data_reader.train_review)
        img_idx = [data_reader.train_id2idx[photo_id] for photo_id in photo_ids]
        images = data_reader.train_img_features[img_idx]

        fd = model.feed_dict(users=review_users, items=review_items, images=images,
                             reviews=reviews, is_training=True)
        _, _review_loss = sess.run([update_review, model.review_loss], feed_dict=fd)
        sum_review_loss += _review_loss

        if _step % FLAGS.display_step == 0:
          data_reader.iter.set_postfix(rating_loss=(sum_rating_loss / count),
                                       review_loss=(sum_review_loss / count))

      # Testing
      review_gen_corpus = defaultdict(list)
      review_ref_corpus = defaultdict(list)

      photo_bleu_scores = defaultdict(list)
      photo_rouge_scores = defaultdict(list)

      review_bleu_scores = defaultdict(list)
      review_rouge_scores = defaultdict(list)

      sess.run(model.init_metrics)
      for users, items, ratings in data_reader.read_test_set(FLAGS.batch_size, rating_only=True):
        test_step += 1

        fd = model.feed_dict(users, items, ratings)
        sess.run(model.update_metrics, feed_dict=fd)

        review_users, review_items, review_ratings, photo_ids, reviews = get_review_data(users, items, ratings,
                                                                                         data_reader.test_review)
        img_idx = [data_reader.test_id2idx[photo_id] for photo_id in photo_ids]
        images = data_reader.test_img_features[img_idx]

        fd = model.feed_dict(users=review_users, items=review_items, images=images)
        _reviews, _alphas, _betas = sess.run([model.sampled_reviews, model.alphas, model.betas], feed_dict=fd)

        gen_reviews = decode_reviews(_reviews, vocab)
        ref_reviews = [decode_reviews(batch_review_normalize(ref), vocab) for ref in reviews]

        for user, item, gen, refs in zip(review_users, review_items, gen_reviews, ref_reviews):
          review_gen_corpus[(user, item)].append(gen)
          review_ref_corpus[(user, item)] += refs

          bleu_scores = compute_bleu([refs], [gen], max_order=4, smooth=True)
          for order, score in bleu_scores.items():
            photo_bleu_scores[order].append(score)

          rouge_scores = rouge([gen], refs)
          for metric, score in rouge_scores.items():
            photo_rouge_scores[metric].append(score)

      _mae, _rmse = sess.run([model.mae, model.rmse])
      log_info(log_file, '\nRating prediction results: MAE={:.3f}, RMSE={:.3f}'.format(_mae, _rmse))

      log_info(log_file, '\nReview generation results:')
      log_info(log_file, '- Photo level: BLEU-scores = {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
        np.array(photo_bleu_scores[1]).mean() * 100, np.array(photo_bleu_scores[2]).mean() * 100,
        np.array(photo_bleu_scores[3]).mean() * 100, np.array(photo_bleu_scores[4]).mean() * 100))

      for user_item, gen_reviews in review_gen_corpus.items():
        references = [list(ref) for ref in set(tuple(ref) for ref in review_ref_corpus[user_item])]

        user_item_bleu_scores = defaultdict(list)
        for gen in gen_reviews:
          bleu_scores = compute_bleu([references], [gen], max_order=4, smooth=True)
          for order, score in bleu_scores.items():
            user_item_bleu_scores[order].append(score)
        for order, scores in user_item_bleu_scores.items():
          review_bleu_scores[order].append(np.array(scores).mean())

        user_item_rouge_scores = defaultdict(list)
        for gen in gen_reviews:
          rouge_scores = rouge([gen], references)
          for metric, score in rouge_scores.items():
            user_item_rouge_scores[metric].append(score)
        for metric, scores in user_item_rouge_scores.items():
          review_rouge_scores[metric].append(np.array(scores).mean())

      log_info(log_file, '- Review level: BLEU-scores = {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
        np.array(review_bleu_scores[1]).mean() * 100, np.array(review_bleu_scores[2]).mean() * 100,
        np.array(review_bleu_scores[3]).mean() * 100, np.array(review_bleu_scores[4]).mean() * 100))

      for metric in ['rouge_1', 'rouge_2', 'rouge_l']:
        log_info(log_file, '- Photo level: {} = {:.2f}, {:.2f}, {:.2f}'.format(
          metric,
          np.array(photo_rouge_scores['{}/p_score'.format(metric)]).mean() * 100,
          np.array(photo_rouge_scores['{}/r_score'.format(metric)]).mean() * 100,
          np.array(photo_rouge_scores['{}/f_score'.format(metric)]).mean() * 100))
        log_info(log_file, '- Review level: {} = {:.2f}, {:.2f}, {:.2f}'.format(
          metric,
          np.array(review_rouge_scores['{}/p_score'.format(metric)]).mean() * 100,
          np.array(review_rouge_scores['{}/r_score'.format(metric)]).mean() * 100,
          np.array(review_rouge_scores['{}/f_score'.format(metric)]).mean() * 100))

      log_info(log_file, '')


if __name__ == '__main__':
  tf.app.run()
