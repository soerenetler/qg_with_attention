import os
import time

import tensorflow as tf

from decoder import Decoder
from encoder import Encoder
from evaluate import QuestionGenerator
from qg_dataset import QGDataset
from utils import convert, generate_embeddings_matrix, loss_function
from bleu_score import BleuScore

#PARAMS
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, default="/content/gdrive/MyDrive/mt-qg-data/01_data/preprocessedData/squad/question_answer/",
                    help="display a square of a given number")
parser.add_argument("-m", "--model", type=str, default="/content/gdrive/MyDrive/mt-qg-data/00_models/qg_attention/squad/",
                    help="display a square of a given number")
parser.add_argument("-g", "--glove", type=str, default="/content/gdrive/MyDrive/mt-qg-data/glove.840B.300d.txt",
                    help="display a square of a given number")
parser.add_argument("-t", "--target_length", type=int, default=20,
                    help="max_length_targ")
parser.add_argument("-i", "--input_length", type=int, default=80,
                    help="display a square of a given number")
parser.add_argument("-x", "--vocab_input", type=int, default=45000,
                    help="display a square of a given number")
parser.add_argument("-y", "--max_vocab_targ", type=int, default=28000,
                    help="display a square of a given number")
parser.add_argument("-e", "--epochs", type=int, default=1,
                    help="display a square of a given number")
parser.add_argument("-u", "--units", type=int, default=600,
                    help="display a square of a given number")
parser.add_argument("-b", "--batch", type=int, default=64,
                    help="display a square of a given number")
args = parser.parse_args()



path_to_folder = args.data
path_to_model = args.model
path_to_glove_file = args.glove
max_length_targ = args.target_length
max_length_inp = args.input_length
max_vocab_inp = args.vocab_input
max_vocab_targ = args.max_vocab_targ
EPOCHS = args.epochs
BATCH_SIZE = args.batch
units = args.units


#SAMPLES
sample_answer_sentence = ['3245', 'two', 'months', 'later', 'the', 'band', 'got', 'signed', 'to', 'a', 'three', 'album', 'deal', 'with', 'spinefarm', ',', 'which', 'left', 'marko', 'displeased', '.']
sample_question_sentence = ['what', 'label', 'were', 'they', 'with', '?']

qg_dataset = QGDataset()
print(qg_dataset.preprocess_sentence(sample_answer_sentence))
print(qg_dataset.preprocess_sentence(sample_question_sentence))

input_tensor_train, target_tensor_train, input_tensor_dev, target_tensor_dev, inp_tokenizer, targ_tokenizer = qg_dataset.load_dataset(max_length_inp=max_length_inp, max_vocab_inp=max_vocab_inp, max_length_targ=max_length_targ, max_vocab_targ=max_vocab_targ)

print("len input_tensor_train: ", len(input_tensor_train))
print("len target_tensor_train: ", len(target_tensor_train))
print("len input_tensor_dev: ", len(input_tensor_dev))
print("len target_tensor_dev", len(target_tensor_dev))

print ("Input Language; index to word mapping")
convert(inp_tokenizer, input_tensor_dev[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_tokenizer, target_tensor_dev[0])

embedding_dim = 300
embedding_matrix = generate_embeddings_matrix(path_to_glove_file, inp_tokenizer, embedding_dim=embedding_dim)

# Create a tf.data dataset
BUFFER_SIZE = len(input_tensor_train)

steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
vocab_inp_size = len(inp_tokenizer.word_index)+1 # PADDING
vocab_tar_size = len(targ_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

dataset_val = tf.data.Dataset.from_tensor_slices((input_tensor_dev, target_tensor_dev))
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print("shape input_batch:", example_input_batch.shape)
print("shape target_batch:", example_target_batch.shape)

example_input_batch_val, example_target_batch_val = next(iter(dataset_val))
print("shape input_batch_val:", example_input_batch_val.shape)
print("shape target_batch_val:", example_target_batch_val.shape)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, bidirectional=False, embedding_matrix=embedding_matrix)
# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, 'luong', max_length_inp=max_length_inp, max_length_targ=max_length_targ)
sample_x = tf.random.uniform((BATCH_SIZE, max_length_targ), dtype=tf.dtypes.float32)
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, sample_hidden, tf.float32)
sample_decoder_outputs = decoder(sample_x, initial_state)
print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

## Define the optimizer and the loss function
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = path_to_model + 'training_checkpoints'
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                 encoder=encoder,
#                                 decoder=decoder)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath=checkpoint_dir + "/mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
]

qg = QuestionGenerator(qg_dataset, inp_tokenizer, encoder, decoder, targ_tokenizer, max_length_inp)
qg.compile(optimizer=optimizer, loss=loss_function)
qg.fit(dataset, epochs=EPOCHS, callbacks=callbacks)

qg.translate(['two', 'months', 'later', 'the', 'band', 'got', 'signed', 'to', 'a', 'three', 'album', 'deal', 'with', ',', 'which', 'left', '.'], attention_plot_folder=path_to_model)
qg.translate(["Golm", "is", "a", "locality", "of", "Potsdam", ",", "the", "capital", "of", "the", "German", "state", "of", "Brandenburg", "."], attention_plot_folder=path_to_model)
qg.translate("the largest of these is the eldon square shop-ping centre , one of the largest city centre shopping com-plexes in the uk .".split(" "), attention_plot_folder=path_to_model)

qg.beam_translate("the largest of these is the eldon square shop-ping centre , one of the largest city centre shopping com-plexes in the uk .".split(" "))
qg.beam_translate(['Golm', 'is', 'a', 'locality', 'of', 'Potsdam', ',', 'the', 'capital', 'of', 'the', 'German', 'state', 'of', 'Brandenburg', '.'])

# dev_sentences, dev_questions = qg_dataset.create_dataset(qg_dataset.dev_path)
# chunks = [dev_sentences[x:x+100] for x in range(0, len(dev_sentences), 100)]
# for chunk in chunks:
#     result, beam_scores = qg.beam_evaluate_sentences(chunk)
#     outputs = qg.targ_tokenizer.sequences_to_texts(result[0])

#     with open(path_to_model + "demo_val.txt", "a") as f:
#         for output in outputs:
#             f.write(str(output))
#             f.write('\n')