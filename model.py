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
            pred = self((inp, targ), training=True, beam_width=1)
            print("TRAIN - targ", targ[0])
            real = targ[:, 1:]         # ignore <start> token
            print("Train - REAL", real[0])

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

    @tf.function
    def test_step(self, data):
        # Unpack the data
        inp, targ = data
        # Compute predictions
        pred = self((inp, targ), training=False, beam_width=3)
        print("TEST - targ", targ[0])
        real = targ[:, 1:]

        print("TEST - REAL", real[0])

        logits = pred.rnn_output
        pred_token = pred.sample_id
        # Updates the metrics tracking the loss
        # self.compiled_loss(real, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(real, pred_token)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, qg_inputs, training=False, beam_width=None):
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
            pred = self.decoder(dec_input, decoder_initial_state, training=True)

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
            length_inp = inp.shape[1]

            enc_start_state = self.encoder.initialize_hidden_state(
                inference_batch_size)
            tf.debugging.assert_shapes([(enc_start_state,(inference_batch_size, self.encoder.enc_units))])
            enc_out, enc_hidden = self.encoder(
                inp, enc_start_state, training=False)

            tf.debugging.assert_shapes([(enc_out, (inference_batch_size, length_inp, self.encoder.enc_units)),
                                        (enc_hidden, (inference_batch_size, self.encoder.enc_units))])
            
            # use teacher forcing
            if beam_width == None:
                raise NotImplementedError("Teacher forcing in evaluation mode not implemented")

            #use GreedyEmbeddingSampler
            elif beam_width == 1:
                # Setup Memory in decoder stack
                self.decoder.attention_mechanism.setup_memory(enc_out)
                # set decoder_initial_state
                pred = self.decoder(None, enc_hidden, training=False)
                return pred

            #use BeamSearch
            elif beam_width > 1:
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
                    self.decoder.rnn_cell, beam_width=beam_width, output_layer=self.decoder.fc, maximum_iterations=30)
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

                return outputs

        else:
            raise NotImplementedError(
                "Call is currently not implemented with training set to {}".format(training))



    def translate(self, sentences, beam_width=1, attention_plot_folder=""):
        proc_sentences = [self.qg_dataset.preprocess_sentence(sentence) for sentence in sentences]

        inputs = self.inp_tokenizer.texts_to_sequences(proc_sentences)
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        if beam_width ==1:
            outputs = self((inputs, None), training=False, beam_width=beam_width).predicted_id.numpy()
        if beam_width > 1:
            outputs = self((inputs, None), training=False, beam_width=beam_width).predicted_id.numpy()
            final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
            beam_scores = tf.transpose(
                outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

            print("final_outputs.shape = (inference_batch_size, beam_width, time_step_outputs) ",
                final_outputs.shape)
            final_outputs[:,0,:]


        print(outputs)
        result_str = self.targ_tokenizer.sequences_to_texts(outputs)
        #attention_matrix = final_state.alignment_history.stack()

        #plot_attention(attention_matrix[:,0,:], proc_sentence, result_str[0].split(" "), folder=attention_plot_folder)
        return result_str
        #print('Input: %s' % (sentence))
        #print('Predicted translation: {}'.format(result_str))

    # def beam_evaluate_sentences(self, inputs, beam_width=3):
    #     inference_batch_size = inputs.shape[0]

    #     hidden = [tf.zeros((inference_batch_size, self.encoder.enc_units))]
    #     enc_out, enc_hidden = self.encoder(inputs, hidden, training=False)

    #     start_tokens = tf.fill([inference_batch_size],
    #                            self.targ_tokenizer.word_index['<start>'])
    #     print("start_tokens.shape: ", start_tokens.shape)
    #     end_token = self.targ_tokenizer.word_index['<end>']
    #     print("end_token: ", end_token)

    #     # From official documentation
    #     # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
    #     # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    #     # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    #     # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

    #     enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
    #     self.decoder.attention_mechanism.setup_memory(enc_out)
    #     print(
    #         "beam_with * [batch_size, max_length_input, rnn_units] :]] :", enc_out.shape)

    #     # set decoder_inital_state which is an AttentionWrapperState considering beam_width
    #     hidden_state = tfa.seq2seq.tile_batch(
    #         enc_hidden, multiplier=beam_width)
    #     decoder_initial_state = self.decoder.rnn_cell.get_initial_state(
    #         batch_size=beam_width*inference_batch_size, dtype=tf.float32)
    #     decoder_initial_state = decoder_initial_state.clone(
    #         cell_state=hidden_state)
    #     print("decoder_initial_state.cell_state: ",
    #           decoder_initial_state.cell_state.shape)
    #     print("decoder_initial_state.attention: ",
    #           decoder_initial_state.attention.shape)
    #     print("decoder_initial_state.alignments: ",
    #           decoder_initial_state.alignments.shape)
    #     print("decoder_initial_state.attention_state: ",
    #           decoder_initial_state.attention_state.shape)

    #     # Instantiate BeamSearchDecoder
    #     decoder_instance = tfa.seq2seq.BeamSearchDecoder(
    #         self.decoder.rnn_cell, beam_width=beam_width, output_layer=self.decoder.fc, maximum_iterations=30)
    #     decoder_embedding_matrix = self.decoder.embedding.variables[0]
    #     print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

    #     # The BeamSearchDecoder object's call() function takes care of everything.
    #     outputs, final_state, sequence_lengths = decoder_instance(
    #         decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
    #     # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object.
    #     # The final beam predictions are stored in outputs.predicted_id
    #     # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
    #     # final_state = tfa.seq2seq.BeamSearchDecoderState object.
    #     # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated
    #     print(
    #         "sequence_lengths.shape = [inference_batch_size, beam_width]: ", sequence_lengths.shape)
    #     print("outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width): ",
    #           outputs.predicted_ids.shape)
    #     print("outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width): ",
    #           outputs.beam_search_decoder_output.scores.shape)
    #     print(type(outputs.beam_search_decoder_output))
    #     print("final_state", final_state)

    #     # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
    #     # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
    #     # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
    #     final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
    #     beam_scores = tf.transpose(
    #         outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

    #     print("final_outputs.shape = (inference_batch_size, beam_width, time_step_outputs) ",
    #           final_outputs.shape)

    #     return final_outputs

    #def beam_translate(self, sentence):
    #    result, beam_scores = self.beam_evaluate_sentences([sentence])
    #    print(result.shape, beam_scores.shape)
    #    for beam, score in zip(result, beam_scores):
    #        print(beam.shape, score.shape)
    #        output = self.targ_tokenizer.sequences_to_texts(beam)
    #        output = [a for a in output]
    #        beam_score = [a.sum() for a in score]
    #        print('Input: %s' % (sentence))
    #        for i in range(len(output)):
    #            print('{} Predicted translation: {}  {}'.format(
    #                i+1, output[i], beam_score[i]))
