import tensorflow as tf
import tensorflow_addons as tfa
from utils import loss_function

class QuestionGenerator(tf.keras.Model):
    def __init__(self, qg_dataset, inp_tokenizer, encoder, decoder, targ_tokenizer, max_length_inp):
        super(QuestionGenerator, self).__init__()
        self.qg_dataset = qg_dataset
        self.inp_tokenizer = inp_tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.targ_tokenizer = targ_tokenizer
        self.max_length_inp = max_length_inp

    ## Training
    @tf.function
    def train_step(self, data):
        inp, targ = data
        loss = 0
        enc_hidden = self.encoder.initialize_hidden_state()
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_input = targ[ : , :-1 ] # Ignore <end> token
            real = targ[ : , 1: ]         # ignore <start> token

            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.decoder.build_initial_state(self.encoder.batch_sz, enc_hidden, tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = loss_function(real, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return {"loss": loss}

    def evaluate_sentence(self, sentence):
        sentence = self.qg_dataset.preprocess_sentence(sentence)

        inputs = self.inp_tokenizer.texts_to_sequences([sentence])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                                maxlen=self.max_length_inp,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        enc_start_state = [tf.zeros((inference_batch_size, self.encoder.enc_units))]
        enc_out, enc_hidden = self.encoder(inputs, enc_start_state, training=False)

        dec_hidden = enc_hidden

        start_tokens = tf.fill([inference_batch_size], self.targ_tokenizer.word_index['<start>'])
        end_token = self.targ_tokenizer.word_index['<end>']

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Instantiate BasicDecoder object
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc, maximum_iterations=10)
        # Setup Memory in decoder stack
        self.decoder.attention_mechanism.setup_memory(enc_out)

        # set decoder_initial_state
        decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, dec_hidden, tf.float32)


        ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
        ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
        ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

        decoder_embedding_matrix = self.decoder.embedding.variables[0]
        print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

        outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
        return outputs.sample_id.numpy()

    def translate(self, sentence):
        result = self.evaluate_sentence(sentence)
        print(result)
        result = self.targ_tokenizer.sequences_to_texts(result)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

    def beam_evaluate_sentence(self, sentence, beam_width=3):
        sentence = self.qg_dataset.preprocess_sentence(sentence)

        inputs = self.inp_tokenizer.texts_to_sequences([sentence])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                                maxlen=self.max_length_inp,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        hidden = [tf.zeros((inference_batch_size, self.encoder.enc_units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden, training=False)

        start_tokens = tf.fill([inference_batch_size], self.targ_tokenizer.word_index['<start>'])
        print("start_tokens.shape: ", start_tokens.shape)
        end_token = self.targ_tokenizer.word_index['<end>']
        print("end_token: ", end_token)

        # From official documentation
        # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
        # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
        # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
        # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

        enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
        self.decoder.attention_mechanism.setup_memory(enc_out)
        print("beam_with * [batch_size, max_length_input, rnn_units] :]] :", enc_out.shape)

        # set decoder_inital_state which is an AttentionWrapperState considering beam_width
        hidden_state = tfa.seq2seq.tile_batch(enc_hidden, multiplier=beam_width)
        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
        decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)
        print("decoder_initial_state.cell_state: ", decoder_initial_state.cell_state.shape)
        print("decoder_initial_state.attention: ", decoder_initial_state.attention.shape)
        print("decoder_initial_state.alignments: ", decoder_initial_state.alignments.shape)
        print("decoder_initial_state.attention_state: ", decoder_initial_state.attention_state.shape)

        # Instantiate BeamSearchDecoder
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(self.decoder.rnn_cell,beam_width=beam_width, output_layer=self.decoder.fc, maximum_iterations=10)
        decoder_embedding_matrix = self.decoder.embedding.variables[0]
        print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

        # The BeamSearchDecoder object's call() function takes care of everything.
        outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
        # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object. 
        # The final beam predictions are stored in outputs.predicted_id
        # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
        # final_state = tfa.seq2seq.BeamSearchDecoderState object.
        # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated
        print("sequence_lengths.shape = [inference_batch_size, beam_width]: ", sequence_lengths.shape)
        print("outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width): ",
                        outputs.predicted_ids.shape)
        print("outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width): ",
                        outputs.beam_search_decoder_output.scores.shape)
        print(type(outputs.beam_search_decoder_output))
        print("final_state", final_state)

        # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
        # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
        # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
        beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))

        print("final_outputs.shape = (inference_batch_size, beam_width, time_step_outputs) ",
                        final_outputs.shape)

        return final_outputs.numpy(), beam_scores.numpy()

    def beam_translate(self, sentence):
        result, beam_scores = self.beam_evaluate_sentence(sentence)
        print(result.shape, beam_scores.shape)
        for beam, score in zip(result, beam_scores):
            print(beam.shape, score.shape)
            output = self.targ_tokenizer.sequences_to_texts(beam)
            output = [a for a in output]
            beam_score = [a.sum() for a in score]
            print('Input: %s' % (sentence))
            for i in range(len(output)):
                print('{} Predicted translation: {}  {}'.format(i+1, output[i], beam_score[i]))

