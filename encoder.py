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


    def call(self, x, training=False):
        x = self.embedding(x)
        result = self.gru(
            x, training=training)
        output = result[0]
        state = tf.concat(result[1:], 1)
        return output, state
