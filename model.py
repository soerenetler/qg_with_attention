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
            #print("TRAIN - targ", targ[0])
            real = targ[:, 1:]         # ignore <start> token
            #print("Train - REAL", real[0])

            #print("TRAIN - pred.rnn_output ", pred.rnn_output.shape)
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
        pred = self((inp, targ), training=False, beam_width=None)
        #print("TEST - targ", targ[0])
        real = targ[:, 1:]

        #print("TEST - REAL", real[0])

        logits = pred.rnn_output
        pred_token = pred.sample_id
        # Updates the metrics tracking the loss
        self.compiled_loss(real, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(real, pred_token)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, qg_inputs, training=False, beam_width=None):
        if training == True or beam_width == None:
            inp, targ = qg_inputs
            # batch_sz = inp.shape[0]
            dec_input = targ[:, :-1]  # Ignore <end> token
            
            #enc_hidden = self.encoder.initialize_hidden_state(batch_sz)
            enc_output, enc_hidden = self.encoder(
                inp, training=training)
            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.decoder.build_initial_state(
                self.decoder.batch_sz, enc_hidden, tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state, training=training, beam_width=None)

            return pred

        elif training == False:
            tf.print("qg_inputs:", qg_inputs)
            if type(qg_inputs) == tf.Tensor:
                inp = qg_inputs
            elif len(qg_inputs) == 2:
                inp, targ = qg_inputs
                tf.print("targ: ", targ)
            else:
                raise NotImplementedError(
                    "Input has a length of {}.".format(len(qg_inputs)))
            # tf.print("INPUT: ", inp)
            # tf.print("INPUT: ", inp)

            #if self.decoder.num_layers ==1:
            inference_batch_size = inp.shape[0]
            length_inp = inp.shape[1]
            # print("model - inference_batch_size:",inference_batch_size)
            #elif self.decoder.num_layers > 1:
            #    tf.print("INPUT[0].shape: ", inp[0].shape)
            #    print("INPUT[0].shape: ", inp[0].shape)
            #    inference_batch_size = inp[0].shape[0]
            #    length_inp = inp[0].shape[1]

            #enc_hidden = self.encoder.initialize_hidden_state(inference_batch_size)

            enc_out, enc_hidden = self.encoder(
                inp, training=False)
            
            if self.encoder.bidirectional:
                if self.decoder.num_layers==1:
                    tf.debugging.assert_shapes([(enc_out, (inference_batch_size, length_inp, self.encoder.enc_units*2)),
                                                (enc_hidden, (inference_batch_size, self.encoder.enc_units*2))])
                if self.decoder.num_layers>1:
                    tf.debugging.assert_shapes([(enc_out, (inference_batch_size, length_inp, self.encoder.enc_units*2)),
                                            (enc_hidden, (self.decoder.num_layers, inference_batch_size, self.encoder.enc_units*2))])  
            else:
                if self.decoder.num_layers==1:
                    tf.debugging.assert_shapes([(enc_out, (inference_batch_size, length_inp, self.encoder.enc_units)),
                                            (enc_hidden, (inference_batch_size, self.encoder.enc_units))])
                if self.decoder.num_layers>1:
                    tf.debugging.assert_shapes([(enc_out, (inference_batch_size, length_inp, self.encoder.enc_units)),
                                                (enc_hidden, (self.decoder.num_layers, inference_batch_size, self.encoder.enc_units))])

            #use GreedyEmbeddingSampler
            if beam_width == 1:
                # Setup Memory in decoder stack
                self.decoder.attention_mechanism.setup_memory(enc_out)
                # set decoder_initial_state
                pred, attention_matrix = self.decoder(None, enc_hidden, training=False, beam_width=1)
                return pred, attention_matrix

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
                #print("decoder_initial_state.cell_state: ",
                #    decoder_initial_state.cell_state.shape)
                #print("decoder_initial_state.attention: ",
                #    decoder_initial_state.attention.shape)
                #print("decoder_initial_state.alignments: ",
                #    decoder_initial_state.alignments.shape)
                #print("decoder_initial_state.attention_state: ",
                #    decoder_initial_state.attention_state.shape)

                # Instantiate BeamSearchDecoder
                decoder_instance = tfa.seq2seq.BeamSearchDecoder(
                    self.decoder.rnn_cell, beam_width=beam_width, output_layer=self.decoder.fc, maximum_iterations=30)
                decoder_embedding_matrix = self.decoder.embedding.variables[0]
                #print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

                # The BeamSearchDecoder object's call() function takes care of everything.
                outputs, final_state, sequence_lengths = decoder_instance(
                    decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
                # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object.
                # The final beam predictions are stored in outputs.predicted_id
                # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
                # final_state = tfa.seq2seq.BeamSearchDecoderState object.
                # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated
                #print(
                #    "sequence_lengths.shape = [inference_batch_size, beam_width]: ", sequence_lengths.shape)
                #print("outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width): ",
                #    outputs.predicted_ids.shape)
                #print("outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width): ",
                #    outputs.beam_search_decoder_output.scores.shape)
                #print(type(outputs.beam_search_decoder_output))
                #print("final_state", final_state)

                # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
                # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
                # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
                final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
                beam_scores = tf.transpose(
                    outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

                tf.debugging.assert_shapes([(final_outputs, (inference_batch_size, beam_width, None))])

                return outputs

        else:
            raise NotImplementedError(
                "Call is currently not implemented with training set to {}".format(training))



    def translate(self, sentences, beam_width=1, attention_plot_folder=""):
        proc_sentences = [self.qg_dataset.preprocess_sentence(sentence, include_eos_bos=False) for sentence in sentences]

        token_inputs = self.inp_tokenizer.texts_to_sequences(proc_sentences)
        pad_inputs = tf.keras.preprocessing.sequence.pad_sequences(token_inputs, padding="post")
                                                             # maxlen=self.max_length_inp,)
        inputs = tf.convert_to_tensor(pad_inputs)

        if beam_width ==1:
            outputs, attention_matrix = self((inputs, None), training=False, beam_width=beam_width)
            outputs = outputs.sample_id.numpy()
            result_str = self.targ_tokenizer.sequences_to_texts(outputs)
            print("attention_matrix.shape: ", attention_matrix.shape)
            for i in range(len(proc_sentences)):
                result_token = [self.targ_tokenizer.index_word[t] for t in outputs[i]]
                input_sentence = [self.inp_tokenizer.index_word[t] for t in token_inputs[i]]
                plot_attention(attention_matrix[:,i,:], input_sentence, result_token, folder=attention_plot_folder)
        if beam_width > 1:
            outputs = self((inputs, None), training=False, beam_width=beam_width)
            final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
            beam_scores = tf.transpose(
                outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

            #print("final_outputs.shape = (inference_batch_size, beam_width, time_step_outputs) ",
            #    final_outputs.shape)
            outputs = final_outputs[:,0,:].numpy()
            result_str = self.targ_tokenizer.sequences_to_texts(outputs)

        print('Input: %s' % (sentences[0]))
        print('Predicted: {}'.format(result_str[0]))
        return result_str
        
