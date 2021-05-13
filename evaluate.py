import tensorflow as tf
import tensorflow_addons as tfa
from utils import plot_attention


class QuestionGenerator(tf.keras.Model):
    def __init__(self, qg_dataset, inp_tokenizer, encoder, decoder, targ_tokenizer, max_length_inp, **kwargs):
        super(QuestionGenerator, self).__init__(**kwargs)
        self.qg_dataset = qg_dataset
        self.inp_tokenizer = inp_tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.targ_tokenizer = targ_tokenizer
        self.max_length_inp = max_length_inp

    # Training
    @tf.function
    def train_step(self, data):
        inp, targ = data

        with tf.GradientTape() as tape:
            pred = self((inp, targ), training=True)
            real = targ[:, 1:]         # ignore <start> token

            print("TRAIN - pred.rnn_output ", pred.rnn_output.shape)
            logits = pred.rnn_output
            pred_token = pred.sample_id
            # Updates the metrics tracking the loss
            loss = self.compiled_loss(
                real, logits, regularization_losses=self.losses)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))
        self.compiled_metrics.update_state(real, pred_token)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        inp, targ = data
        # Compute predictions
        pred = self((inp, None), training=False)
        real = targ[:, 1:]
        logits = pred.rnn_output
        pred_token = pred.sample_id
        # Updates the metrics tracking the loss
        self.compiled_loss(real, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(real, pred_token)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, qg_inputs, training=None):
        if training == True:
            inp, targ = qg_inputs
            batch_sz = inp.shape[0]

            dec_input = targ[:, :-1]  # Ignore <end> token

            enc_hidden = self.encoder.initialize_hidden_state(batch_sz)
            enc_output, enc_hidden = self.encoder(
                inp, enc_hidden, training=training)
            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.decoder.build_initial_state(
                self.encoder.batch_sz, enc_hidden, tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state)

            return pred
        elif training == False:
            print("qg_inputs:", qg_inputs)
            if type(qg_inputs) == tf.Tensor:
                inp = qg_inputs
            elif len(qg_inputs) == 2:
                inp, _ = qg_inputs
            else:
                raise NotImplementedError(
                    "Input has a length of {}.".format(len(qg_inputs)))

            inference_batch_size = inp.shape[0]

            enc_start_state = self.encoder.initialize_hidden_state(
                inference_batch_size)
            enc_out, enc_hidden = self.encoder(
                inp, enc_start_state, training=False)

            # Setup Memory in decoder stack
            self.decoder.attention_mechanism.setup_memory(enc_out)
            # set decoder_initial_state
            decoder_initial_state = self.decoder.build_initial_state(
                inference_batch_size, enc_hidden, tf.float32)

            self.decoder(None, decoder_initial_state, training=False)

        else:
            raise NotImplementedError(
                "Call is currently not implemented with training set to {}".format(training))

    def evaluate_sentence(self, sentence):
        proc_sentence = self.qg_dataset.preprocess_sentence(sentence)

        inputs = self.inp_tokenizer.texts_to_sequences([proc_sentence])
        # inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
        #                                                        maxlen=self.max_length_inp,
        #                                                        padding='post')
        inputs = tf.convert_to_tensor(inputs)
        outputs = self((inputs, None), training=False)

        #inference_batch_size = inputs.shape[0]

        #enc_start_state = [tf.zeros((inference_batch_size, self.encoder.enc_units))]
        #enc_out, enc_hidden = self.encoder(inputs, enc_start_state, training=False)

        #dec_hidden = enc_hidden

        #start_tokens = tf.fill([inference_batch_size], self.targ_tokenizer.word_index['<start>'])
        #end_token = self.targ_tokenizer.word_index['<end>']

        #greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Instantiate BasicDecoder object
        #decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc, maximum_iterations=20)
        # Setup Memory in decoder stack
        # self.decoder.attention_mechanism.setup_memory(enc_out)

        # set decoder_initial_state
        #decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, dec_hidden, tf.float32)

        # Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
        # decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
        # You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

        #decoder_embedding_matrix = self.decoder.embedding.variables[0]
        #print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

        #outputs, final_state, sequence_lengths= decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
        #print("final_state, ", final_state)
        #print("final_state.alignment_history, ", final_state.alignment_history)
        #print("final_state.alignment_history.stack(), ", final_state.alignment_history.stack())

        return outputs, None, proc_sentence

    def translate(self, sentence, attention_plot_folder=""):
        result, final_state, proc_sentence = self.evaluate_sentence(sentence)
        print(result)
        result_str = self.targ_tokenizer.sequences_to_texts(
            result.sample_id.numpy())
        #attention_matrix = final_state.alignment_history.stack()

        #plot_attention(attention_matrix[:,0,:], proc_sentence, result_str[0].split(" "), folder=attention_plot_folder)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result_str))

    def beam_evaluate_sentences(self, sentences, beam_width=3):
        sentences = [self.qg_dataset.preprocess_sentence(
            sentence) for sentence in sentences]

        inputs = self.inp_tokenizer.texts_to_sequences(sentences)
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        hidden = [tf.zeros((inference_batch_size, self.encoder.enc_units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden, training=False)

        start_tokens = tf.fill([inference_batch_size],
                               self.targ_tokenizer.word_index['<start>'])
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
        print(
            "beam_with * [batch_size, max_length_input, rnn_units] :]] :", enc_out.shape)

        # set decoder_inital_state which is an AttentionWrapperState considering beam_width
        hidden_state = tfa.seq2seq.tile_batch(
            enc_hidden, multiplier=beam_width)
        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(
            batch_size=beam_width*inference_batch_size, dtype=tf.float32)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=hidden_state)
        print("decoder_initial_state.cell_state: ",
              decoder_initial_state.cell_state.shape)
        print("decoder_initial_state.attention: ",
              decoder_initial_state.attention.shape)
        print("decoder_initial_state.alignments: ",
              decoder_initial_state.alignments.shape)
        print("decoder_initial_state.attention_state: ",
              decoder_initial_state.attention_state.shape)

        # Instantiate BeamSearchDecoder
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(
            self.decoder.rnn_cell, beam_width=beam_width, output_layer=self.decoder.fc, maximum_iterations=20)
        decoder_embedding_matrix = self.decoder.embedding.variables[0]
        print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

        # The BeamSearchDecoder object's call() function takes care of everything.
        outputs, final_state, sequence_lengths = decoder_instance(
            decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
        # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object.
        # The final beam predictions are stored in outputs.predicted_id
        # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
        # final_state = tfa.seq2seq.BeamSearchDecoderState object.
        # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated
        print(
            "sequence_lengths.shape = [inference_batch_size, beam_width]: ", sequence_lengths.shape)
        print("outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width): ",
              outputs.predicted_ids.shape)
        print("outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width): ",
              outputs.beam_search_decoder_output.scores.shape)
        print(type(outputs.beam_search_decoder_output))
        print("final_state", final_state)

        # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
        # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
        # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
        beam_scores = tf.transpose(
            outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

        print("final_outputs.shape = (inference_batch_size, beam_width, time_step_outputs) ",
              final_outputs.shape)

        return final_outputs.numpy(), beam_scores.numpy()

    def beam_translate(self, sentence):
        result, beam_scores = self.beam_evaluate_sentences([sentence])
        print(result.shape, beam_scores.shape)
        for beam, score in zip(result, beam_scores):
            print(beam.shape, score.shape)
            output = self.targ_tokenizer.sequences_to_texts(beam)
            output = [a for a in output]
            beam_score = [a.sum() for a in score]
            print('Input: %s' % (sentence))
            for i in range(len(output)):
                print('{} Predicted translation: {}  {}'.format(
                    i+1, output[i], beam_score[i]))
