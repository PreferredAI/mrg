import tensorflow as tf

from reader import batch_review_normalize, PAD_INDEX, START_INDEX
from utils import load_glove, get_shape


class Model:

  def __init__(self, total_users, total_items, global_rating,
               num_factors, img_dims, vocab_size,
               word_dim, lstm_dim, max_length, dropout_rate):
    self.total_users = total_users
    self.total_items = total_items
    self.global_rating = global_rating
    self.dropout_rate = dropout_rate
    self.F = num_factors
    self.L = img_dims[0]
    self.D = img_dims[1]
    self.V = vocab_size
    self.W = word_dim
    self.C = lstm_dim
    self.T = max_length

    self.weight_initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer = tf.zeros_initializer()

    self.users = tf.placeholder(tf.int32, shape=[None])
    self.items = tf.placeholder(tf.int32, shape=[None])
    self.ratings = tf.placeholder(tf.float32, shape=[None])
    self.images = tf.placeholder(tf.float32, shape=[None, self.L, self.D])
    self.reviews = tf.placeholder(tf.int32, shape=[None, None])
    self.is_training = tf.placeholder(tf.bool)

    self._init_embeddings()

    self.user_emb = tf.nn.embedding_lookup(self.user_matrix, self.users)
    self.item_emb = tf.nn.embedding_lookup(self.item_matrix, self.items)

    self.sentiment_features = self._get_features(self.user_emb, self.item_emb)
    self.sentiment_features = self._batch_norm(self.sentiment_features, name='review/sentiment')
    self.visual_features = self._batch_norm(self.images, name='review/visual')
    self.visual_projection = self._visual_projection(self.visual_features)

    self._build_rating_predictor()
    self._build_review_generator()
    self._build_review_sampler(max_decode_length=self.T)

  def _init_embeddings(self):
    self.user_matrix = tf.get_variable(
      name='user_matrix',
      shape=[self.total_users, self.F],
      initializer=self.weight_initializer,
      dtype=tf.float32
    )

    self.item_matrix = tf.get_variable(
      name='item_matrix',
      shape=[self.total_items, self.F],
      initializer=self.weight_initializer,
      dtype=tf.float32
    )

    self.word_matrix = tf.get_variable(
      name='word_matrix',
      shape=[self.V, self.W],
      initializer=tf.constant_initializer(load_glove(self.V, self.W)),
      dtype=tf.float32
    )

  def _get_features(self, user_emb, item_emb, num_layers=1):
    with tf.variable_scope('features', reuse=tf.AUTO_REUSE):
      features = tf.concat([user_emb, item_emb], axis=1)
      for layer in range(num_layers):
        w = tf.get_variable('w{}'.format(layer), [2 * self.F, 2 * self.F], initializer=self.weight_initializer)
        b = tf.get_variable('b{}'.format(layer), [2 * self.F], initializer=self.const_initializer)
        features = tf.matmul(features, w) + b
        features = tf.nn.tanh(features, 'h{}'.format(layer))

      return features

  def _build_rating_predictor(self):
    features = self._get_features(self.user_emb, self.item_emb)

    with tf.variable_scope('rating'):
      rating_labels = tf.reshape(self.ratings, [-1, 1])
      rating_preds = self.global_rating + tf.layers.dense(features, units=1, name='prediction')
      self.rating_loss = tf.losses.mean_squared_error(rating_labels, rating_preds)

      self.rating_preds = tf.clip_by_value(rating_preds, clip_value_min=1.0, clip_value_max=5.0)
      self.mae, mae_update = tf.metrics.mean_absolute_error(rating_labels, self.rating_preds, name='metrics/MAE')
      self.rmse, rmse_update = tf.metrics.root_mean_squared_error(rating_labels, self.rating_preds, name='metrics/RMSE')

      metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="rating/metrics")
      self.init_metrics = tf.variables_initializer(var_list=metric_vars)
      self.update_metrics = tf.group([mae_update, rmse_update])

  def _batch_norm(self, x, name=None):
    return tf.contrib.layers.batch_norm(inputs=x,
                                        decay=0.95,
                                        center=True,
                                        scale=True,
                                        is_training=self.is_training,
                                        updates_collections=None,
                                        scope=(name + '_batch_norm'))

  def _visual_projection(self, features):
    with tf.variable_scope('review/visual_projection'):
      w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
      features_flat = tf.reshape(features, [-1, self.D])
      features_proj = tf.matmul(features_flat, w)
      features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
      return features_proj

  def _attention_layer(self, h, features, features_proj):
    with tf.variable_scope('attention'):
      L = get_shape(features)[1]

      w = tf.get_variable('w', [self.C, self.D], initializer=self.weight_initializer)
      b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
      w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)
      b_att = tf.get_variable('b_att', [1], initializer=self.const_initializer)

      h_att = tf.nn.tanh(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)
      out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att) + b_att, [-1, L])
      alpha = tf.nn.softmax(out_att)
      context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')

    return context, alpha

  def _fusion_gate(self, x, h, s_features, v_features):
    with tf.variable_scope('fusion_gate'):
      w_x = tf.get_variable('w_x', [self.W, 1], initializer=self.weight_initializer)
      w_h = tf.get_variable('w_h', [self.C, 1], initializer=self.weight_initializer)
      b = tf.get_variable('b', [1], initializer=self.const_initializer)
      beta = tf.nn.sigmoid(tf.matmul(x, w_x) + tf.matmul(h, w_h) + b)  # (N, 1)
      weighted_features = tf.multiply(beta, s_features) + tf.multiply((1. - beta), v_features)
      return weighted_features, beta

  def _init_lstm(self):
    with tf.variable_scope('init_lstm'):
      user_item_emb = tf.concat([self.user_emb, self.item_emb], axis=1)

      w_h_ui = tf.get_variable('w_h_ui', [self.D, self.C], initializer=self.weight_initializer)
      b_h = tf.get_variable('b_h', [self.C], initializer=self.const_initializer)
      h = tf.matmul(user_item_emb, w_h_ui) + b_h

      w_c_ui = tf.get_variable('w_c_ui', [self.D, self.C], initializer=self.weight_initializer)
      b_c = tf.get_variable('b_c', [self.C], initializer=self.const_initializer)
      c = tf.matmul(user_item_emb, w_c_ui) + b_c

      h = tf.nn.tanh(h)
      c = tf.nn.tanh(c)
      return c, h

  def _decode_lstm(self, x, h, context):
    with tf.variable_scope('decode_lstm'):
      w_h = tf.get_variable('w_h', [self.C, self.W], initializer=self.weight_initializer)
      b_h = tf.get_variable('b_h', [self.W], initializer=self.const_initializer)
      w_out = tf.get_variable('w_out', [self.W, self.V], initializer=self.weight_initializer)
      b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

      h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
      h_logits = tf.matmul(h, w_h) + b_h

      w_ctx2out = tf.get_variable('w_ctx2out', [get_shape(context)[1], self.W], initializer=self.weight_initializer)
      h_logits += tf.matmul(context, w_ctx2out)

      h_logits += x
      h_logits = tf.nn.tanh(h_logits)

      h_logits = tf.layers.dropout(h_logits, self.dropout_rate, training=self.is_training)
      out_logits = tf.matmul(h_logits, w_out) + b_out
      return out_logits

  def _build_review_generator(self):
    with tf.variable_scope('review', reuse=tf.AUTO_REUSE):
      reviews_inputs = self.reviews[:, :self.T - 1]
      reviews_emb = tf.nn.embedding_lookup(self.word_matrix, reviews_inputs)
      reviews_labels = self.reviews[:, 1:]
      mask = tf.to_float(tf.not_equal(reviews_labels, PAD_INDEX))

      loss = 0.0

      self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.C, name='LSTM_Cell')
      c, h = self._init_lstm()

      for t in range(self.T - 1):
        x = reviews_emb[:, t, :]

        visual_context, alpha = self._attention_layer(h, self.visual_features, self.visual_projection)
        context, beta = self._fusion_gate(x, h, self.sentiment_features, visual_context)

        cell_input = tf.concat([x, context], axis=1)
        _, (c, h) = self.cell(inputs=cell_input, state=[c, h])

        logits = self._decode_lstm(x, h, context)

        loss += tf.reduce_sum(
          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reviews_labels[:, t], logits=logits) * mask[:, t])

      self.review_loss = loss / tf.reduce_sum(mask)

  def _build_review_sampler(self, max_decode_length):
    with tf.variable_scope('review', reuse=tf.AUTO_REUSE):
      sampled_word_list = []
      beta_list = []
      alpha_list = []

      c, h = self._init_lstm()

      batch_size = tf.shape(self.users)[0]
      sampled_word = tf.fill([batch_size], START_INDEX)
      for t in range(max_decode_length):
        x = tf.nn.embedding_lookup(self.word_matrix, sampled_word)

        visual_context, alpha = self._attention_layer(h, self.visual_features, self.visual_projection)
        alpha_list.append(alpha)
        context, beta = self._fusion_gate(x, h, self.sentiment_features, visual_context)
        beta_list.append(beta)

        cell_input = tf.concat([x, context], axis=1)
        _, (c, h) = self.cell(inputs=cell_input, state=[c, h])

        logits = self._decode_lstm(x, h, context)

        sampled_word = tf.argmax(logits, 1)
        sampled_word_list.append(sampled_word)

      self.sampled_reviews = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (N, max_len)
      self.alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
      self.betas = tf.transpose(tf.squeeze(tf.stack(beta_list), axis=2), (1, 0))  # (N, T)

  def feed_dict(self, users, items, ratings=None, images=None, reviews=None, is_training=False):
    fd = {
      self.users: users,
      self.items: items,
      self.is_training: is_training
    }
    if ratings is not None:
      fd[self.ratings] = ratings
    if images is not None:
      fd[self.images] = images
    if reviews is not None:
      fd[self.reviews] = batch_review_normalize(reviews, self.T)

    return fd
