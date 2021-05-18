import datetime
import os
import time
from unicodedata import bidirectional

import tensorflow as tf
import shutil

from decoder import Decoder
from encoder import Encoder
from model import QuestionGenerator
from qg_dataset import QGDataset
from utils import convert, generate_embeddings_matrix, loss_function
#from bleu_score import BleuScore

# PARAMS
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="squad",
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
parser.add_argument("-l", "--layer", type=int, default=1,
                    help="display a square of a given number")
parser.add_argument("-o", "--dropout", type=float, default=0.4,
                    help="display a square of a given number")
args = parser.parse_args()

print(args)
print(type(args))

modelname = ""
for key, value in vars(args).items():
    modelname += key + "_" + str(value) + "-"

path_to_folder = "/content/gdrive/MyDrive/mt-qg-data/01_data/preprocessedData/" + \
    args.dataset + "/question_answer/"
path_to_model = "/content/gdrive/MyDrive/mt-qg-data/00_models/qg_attention/" + \
    args.dataset + "/" + modelname + "/"


try:
    shutil.rmtree(path_to_model)
except FileNotFoundError:
    print("Directory does not excist: {}".format(path_to_model))

path_to_logs = "/content/gdrive/MyDrive/mt-qg-data/02_logs/qg_attention/" + \
    args.dataset + "/" + modelname + "/"

try:
    shutil.rmtree(path_to_logs)
except FileNotFoundError:
    print("Directory does not excist: {}".format(path_to_logs))


path_to_glove_file = "/content/gdrive/MyDrive/mt-qg-data/glove.840B.300d.txt"
max_length_targ = args.target_length
max_length_inp = args.input_length
max_vocab_inp = args.vocab_input
max_vocab_targ = args.max_vocab_targ
layer = args.layer
EPOCHS = args.epochs
BATCH_SIZE = args.batch
units = args.units
dropout = args.dropout


# SAMPLES
sample_answer_sentence = ['3245', 'two', 'months', 'later', 'the', 'band', 'got', 'signed', 'to', 'a',
                          'three', 'album', 'deal', 'with', 'spinefarm', ',', 'which', 'left', 'marko', 'displeased', '.']
sample_question_sentence = ['what', 'label', 'were', 'they', 'with', '?']

qg_dataset = QGDataset()
print(qg_dataset.preprocess_sentence(sample_answer_sentence))
print(qg_dataset.preprocess_sentence(sample_question_sentence))

input_tensor_train, target_tensor_train, input_tensor_dev, target_tensor_dev, inp_tokenizer, targ_tokenizer = qg_dataset.load_dataset(
    max_length_inp=max_length_inp, max_vocab_inp=max_vocab_inp, max_length_targ=max_length_targ, max_vocab_targ=max_vocab_targ)

print("len input_tensor_train: ", len(input_tensor_train))
print("len target_tensor_train: ", len(target_tensor_train))
print("len input_tensor_dev: ", len(input_tensor_dev))
print("len target_tensor_dev", len(target_tensor_dev))

print("Input Language; index to word mapping")
convert(inp_tokenizer, input_tensor_dev[0])
print()
print("Target Language; index to word mapping")
convert(targ_tokenizer, target_tensor_dev[0])

embedding_dim = 300
inp_embedding_matrix = generate_embeddings_matrix(
    path_to_glove_file, inp_tokenizer, embedding_dim=embedding_dim)

targ_embedding_matrix = generate_embeddings_matrix(
    path_to_glove_file, targ_tokenizer, embedding_dim=embedding_dim)

# Create a tf.data dataset
BUFFER_SIZE = len(input_tensor_train)

steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
vocab_inp_size = len(inp_tokenizer.word_index)+1  # PADDING
vocab_tar_size = len(targ_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

dataset_val = tf.data.Dataset.from_tensor_slices(
    (input_tensor_dev, target_tensor_dev))
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
tf.debugging.assert_shapes(
    [(example_input_batch, (BATCH_SIZE, max_length_inp))])
tf.debugging.assert_shapes(
    [(example_target_batch, (BATCH_SIZE, max_length_targ))])

example_input_batch_val, example_target_batch_val = next(iter(dataset_val))
tf.debugging.assert_shapes(
    [(example_input_batch_val, (BATCH_SIZE, max_length_inp))])
tf.debugging.assert_shapes(
    [(example_target_batch_val, (BATCH_SIZE, max_length_targ))])

encoder = Encoder(vocab_inp_size, embedding_dim, units,
                  bidirectional=True, embedding_matrix=inp_embedding_matrix, layer=layer, dropout=dropout)
# sample input
sample_output, sample_hidden = encoder(
    example_input_batch, training=True)
tf.debugging.assert_shapes(
    [(sample_output, (BATCH_SIZE, max_length_inp, units))])

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE,
                  targ_tokenizer.word_index['<start>'], targ_tokenizer.word_index['<end>'],  attention_type='luong', max_length_inp=max_length_inp, max_length_targ=max_length_targ, embedding_matrix=targ_embedding_matrix, layer=layer, dropout=dropout)
sample_x = tf.random.uniform(
    (BATCH_SIZE, max_length_targ), dtype=tf.dtypes.float32)
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(
    BATCH_SIZE, sample_hidden, tf.float32)
sample_decoder_outputs = decoder(sample_x, initial_state, training=True)
tf.debugging.assert_shapes(
    [(sample_decoder_outputs.rnn_output, (BATCH_SIZE, max_length_targ-1, vocab_tar_size))])

# Define the optimizer and the loss function
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = path_to_model + 'training_checkpoints'

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=path_to_logs, histogram_freq=1)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/model_{epoch}",
                                                         # save_best_only=True,  # Only save a model if `val_loss` has improved.
                                                         # monitor="val_loss",
                                                         verbose=1,
                                                         )

qg = QuestionGenerator(qg_dataset, inp_tokenizer, encoder,
                       decoder, targ_tokenizer, max_length_inp)
qg.compile(optimizer=optimizer, loss=loss_function)
# qg.build(tf.TensorShape((BATCH_SIZE, max_length_inp)))
# qg.summary()
qg.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[
       checkpoint_callback, tensorboard_callback], validation_data=dataset_val)

# qg.save(path_to_model+"saved_model/")

qg.translate(['two', 'months', 'later', 'the', 'band', 'got', 'signed', 'to', 'a', 'three',
             'album', 'deal', 'with', ',', 'which', 'left', '.'], attention_plot_folder=path_to_model)
qg.translate(["Golm", "is", "a", "locality", "of", "Potsdam", ",", "the", "capital", "of",
             "the", "German", "state", "of", "Brandenburg", "."], attention_plot_folder=path_to_model)
qg.translate("the largest of these is the eldon square shop-ping centre , one of the largest city centre shopping com-plexes in the uk .".split(" "),
             attention_plot_folder=path_to_model)

qg.translate(
    "the largest of these is the eldon square shop-ping centre , one of the largest city centre shopping com-plexes in the uk .".split(" "), beam_width=3)
qg.translate(['Golm', 'is', 'a', 'locality', 'of', 'Potsdam', ',', 'the',
              'capital', 'of', 'the', 'German', 'state', 'of', 'Brandenburg', '.'], beam_width=3)

dev_sentences, dev_questions = qg_dataset.create_dataset(qg_dataset.dev_path)
chunks = [dev_sentences[x:x+100] for x in range(0, len(dev_sentences), 100)]
for chunk in chunks:
    outputs = qg.translate(chunk, beam_width=3)

    filename = modelname + ".txt"

    with open(path_to_model + filename, "a") as f:
        for output in outputs:
            f.write(str(output))
            f.write('\n')
