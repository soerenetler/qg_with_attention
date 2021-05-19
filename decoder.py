import tensorflow as tf
import tensorflow_addons as tfa


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, start_token, end_token, layer=1, attention_type='luong', max_length_inp=80, max_length_targ=20, embedding_matrix=None,dropout=0.3, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_inp = max_length_inp
        self.max_length_targ = max_length_targ
        self.start_token = start_token
        self.end_token = end_token
        self.layer = layer

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   embeddings_initializer=tf.keras.initializers.Constant(
                                                       embedding_matrix),
                                                   trainable=False)

        # Define the fundamental cell for decoder recurrent structure
        if self.layer == 1:
            self.gru = tf.keras.layers.GRUCell(self.dec_units,
                                               recurrent_initializer='glorot_uniform',
                                               dropout=dropout)
        elif self.layer > 1:
            rnn_cells = [tf.keras.layers.GRUCell(
                self.dec_units, dropout=dropout) for _ in range(self.layer)]
            self.gru = tf.keras.layers.StackedRNNCells(rnn_cells)
        else:
            raise NotImplementedError(
                "Number of Layer is not implemented: {}".format(self.layer))

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Sampler
        self.train_sampler = tfa.seq2seq.TrainingSampler()
        self.inference_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                                  None, self.batch_sz*[self.max_length_inp], self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(self.gru,
                                                     self.attention_mechanism, attention_layer_size=self.dec_units, alignment_history=True)

        # Define the decoder with respect to fundamental rnn cell
        self.train_decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.train_sampler, output_layer=self.fc)
        # Instantiate BasicDecoder object
        self.inference_decoder = tfa.seq2seq.BasicDecoder(
            cell=self.rnn_cell, sampler=self.inference_sampler, output_layer=self.fc, maximum_iterations=30)

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if(attention_type == 'bahdanau'):
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)
        return decoder_initial_state

    def call(self, x, hidden, training=False, beam_width=None):
        if training == True or beam_width==None:
            print("initial_state: ", hidden)
            print("dec_input: ", x)
            x = self.embedding(x)
            outputs, _, _ = self.train_decoder(
                x, initial_state=hidden, sequence_length=self.batch_sz*[self.max_length_targ-1])
            return outputs
        elif training == False:
            if self.layer ==1:
              inference_batch_size = hidden.shape[0]
            elif self.layer > 1:
              inference_batch_size = hidden[0].shape[0]
            print("Decoder - inference_batch_size:",inference_batch_size)
            decoder_initial_state = self.build_initial_state(
                inference_batch_size, hidden, tf.float32)
            start_tokens = tf.fill(
                [inference_batch_size], self.start_token)
            decoder_embedding_matrix = self.embedding.variables[0]
            print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

            outputs, final_state, sequence_lengths = self.inference_decoder(
                decoder_embedding_matrix, start_tokens=start_tokens, end_token=self.end_token, initial_state=decoder_initial_state)
            print("sequence_lengths", sequence_lengths)
            print("final_state, ", final_state)
            print("final_state.alignment_history, ",
                  final_state.alignment_history)
            print("final_state.alignment_history.stack(), ",
                  final_state.alignment_history.stack())
            print("EVALUATION - Outputs", outputs.sample_id.shape)
            return outputs
        else:
            raise NotImplementedError(
                "Call is currently not implemented with training set to {}".format(training))
