import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, embedding_matrix=None, pretraine_embeddings=False, bidirectional=False, layer=1, dropout=0.4, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.bidirectional = bidirectional
        self.layer = layer

        if self.bidirectional:
            self.enc_units = int(enc_units/2)
        else:
            self.enc_units = enc_units

        if pretraine_embeddings:
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       embedding_dim,
                                                       embeddings_initializer=tf.keras.initializers.Constant(
                                                           embedding_matrix),
                                                       trainable=False, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       embedding_dim,
                                                       trainable=True, mask_zero=True)


        def init_layer():
            if bidirectional:
                return tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,))
                #stateful=True)

            else:
                return tf.keras.layers.GRU(
                    self.enc_units,
                    return_sequences=True,
                    return_state=True,)
                    #stateful=True)

        self._layer_names = ['layer_' + str(i) for i in range(self.layer)]
        for name in self._layer_names:
             self.__setattr__(name, init_layer())
        

        #if self.layer == 0:
        #    self.gru = tf.keras.layers.GRU(self.enc_units,
        #                                   return_sequences=True,
        #                                   return_state=True,
        #                                   dropout=dropout, recurrent_dropout=dropout)
        #elif self.layer > 0:
        #    rnn_cells = [tf.keras.layers.GRUCell(
        #        self.enc_units, dropout=dropout, recurrent_dropout=dropout) for _ in range(self.layer)]
        #    stacked_gru = tf.keras.layers.StackedRNNCells(rnn_cells)
        #    self.gru = tf.keras.layers.RNN(stacked_gru, return_sequences=True,
        #                                   return_state=True)
        #else:
        #    raise NotImplementedError(
        #        "Layer in encoder: {}".format(self.layer))
        #if self.bidirectional:
        #    self.gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, x, training=False):
        x = self.embedding(x,training=training)

        seqs = x
        states = []
        for name in self._layer_names:
            rnn = self.__getattribute__(name)
            output = rnn(seqs)
            if self.bidirectional:
                seqs = output[0]
                state = tf.concat(output[1:],1)
            else:
                (seqs, state) = output
            states.append(state)

        print("Encoder result:", seqs.shape)
        for i, state in enumerate(states):
            print("Encoder State", i, state.shape)

        if len(states) == 1:
            return seqs, states[0]
        else:
            return seqs, tuple(states)       

#    def initialize_hidden_state(self, batch_sz):
#        if self.bidirectional:
#            return [tf.zeros((batch_sz, self.enc_units)) for i in range(2)]
#        else:
#            return tf.zeros((batch_sz, self.enc_units))
