import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix, bidirectional=False, layer=1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.bidirectional = bidirectional
        self.layer = layer

        if self.bidirectional:
            self.enc_units = int(enc_units/2)
        else:
            self.enc_units = enc_units
        self.batch_sz = batch_sz

        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   embeddings_initializer=tf.keras.initializers.Constant(
                                                       embedding_matrix),
                                                   trainable=False)

        if self.layer == 1:
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=0.3)
        elif self.layer >1:
            rnn_cells = [tf.keras.layers.GRUCell(self.enc_units, dropout=0.3) for _ in range(2)]
            stacked_gru = tf.keras.layers.StackedRNNCells(rnn_cells)
            self.gru = tf.keras.layers.RNN(stacked_gru, return_sequences=True,
                                        return_state=True)
        else:
            raise NotImplementedError("Layer in encoder: {}".format(self.layer))
        

        if self.bidirectional:
            self.gru = tf.keras.layers.Bidirectional(self.gru)


    def call(self, x, hidden, training=False):
        x = self.embedding(x)
        if self.bidirectional:
            output, forward_state, backward_state = self.gru(
                x, initial_state=hidden, training=training)
            state = tf.concat([forward_state, backward_state], 1)
        else:
            output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_sz):
        if self.bidirectional:
            return [tf.zeros((batch_sz, self.enc_units)) for i in range(2)]
        else:
            return tf.zeros((batch_sz, self.enc_units))
