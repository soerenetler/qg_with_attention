import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, bidirectional=False, embedding_matrix=None):
    super(Encoder, self).__init__()
    self.bidirectional=bidirectional
    self.batch_sz = batch_sz
    self.enc_units = enc_units

    if embedding_matrix != None:
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                embedding_dim,
                                                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                trainable=False)
    else:
        # TODO maybe implement encoder without pretrained embeddings
        raise ValueError("embedding_matrix should not be none")
    
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout=0.3) # TODO I need to make this bidirectional
    if self.bidirectional:
      self.gru = tf.keras.layers.Bidirectional(self.gru)


  def call(self, x, hidden, training=True):
    x = self.embedding(x)
    if self.bidirectional:
      output, forward_state, backward_state = self.gru(x, initial_state = hidden, training= training)
      state = tf.concat([forward_state, backward_state], 1)
    else:
      output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    if self.bidirectional:
      return [tf.zeros((self.batch_sz, self.enc_units)) for i in range(2)]
    else:
      return tf.zeros((self.batch_sz, self.enc_units))