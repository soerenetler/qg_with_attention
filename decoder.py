import tensorflow as tf
import tensorflow_addons as tfa

class Decoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong', max_length_inp=80, max_length_targ=20, **kwargs):
    super(Decoder, self).__init__(**kwargs)
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type
    self.max_length_inp=max_length_inp
    self.max_length_targ=max_length_targ

    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    # Define the fundamental cell for decoder recurrent structure
    self.gru = tf.keras.layers.GRUCell(self.dec_units,
                                       recurrent_initializer='glorot_uniform')
    # TODO MAKE THIS 2 LAYER???
    
    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Sampler
    self.train_sampler = tfa.seq2seq.sampler.TrainingSampler()
    self.inference_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[self.max_length_inp], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = tfa.seq2seq.AttentionWrapper(self.gru, 
                                  self.attention_mechanism, attention_layer_size=self.dec_units, alignment_history=True)
    
    # Define the decoder with respect to fundamental rnn cell
    self.train_decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.train_sampler, output_layer=self.fc)
    # Instantiate BasicDecoder object
    self.inference_decoder = tfa.seq2seq.BasicDecoder(cell=self.rnn_cell, sampler=self.inference_sampler, output_layer=self.fc, maximum_iterations=max_length_targ)

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    # ------------- #
    # typ: Which sort of attention (Bahdanau, Luong)
    # dec_units: final dimension of attention outputs 
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state

  def call(self, x, hidden):
    print("initial_state: ", hidden)
    print("dec_input: ", x)
    x = self.embedding(x)    
    outputs, _, _ = self.train_decoder(x, initial_state=hidden, sequence_length=self.batch_sz*[self.max_length_targ-1])
    return outputs