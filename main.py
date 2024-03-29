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
from distutils.util import strtobool
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
parser.add_argument("-o", "--dropout", type=float, default=0.3,
                    help="display a square of a given number")
parser.add_argument("-p", "--pretrained", type=lambda x: bool(strtobool(x)), default=False,
                    help="display a square of a given number")
parser.add_argument("-r", "--bidirectional", type=lambda x: bool(strtobool(x)), default=True,
                    help="display a square of a given number")
parser.add_argument("-a", "--answer_units", type=int, default=0,
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
pretrained = args.pretrained
bidirectional = args.bidirectional
answer_enc_units = args.answer_units


# SAMPLES
sample_answer_sentence = ['3245', 'two', 'months', 'later', 'the', 'band', 'got', 'signed', 'to', 'a',
                          'three', 'album', 'deal', 'with', 'spinefarm', ',', 'which', 'left', 'marko', 'displeased', '.']
sample_question_sentence = ['what', 'label', 'were', 'they', 'with', '?']
sample_answer = ['spinefarm']

qg_dataset = QGDataset(problem_type=args.dataset)
print(qg_dataset.preprocess_sentence(sample_answer_sentence))
print(qg_dataset.preprocess_sentence(sample_question_sentence))
print(qg_dataset.preprocess_sentence(sample_answer))

ans_sent_tensor_train, ans_token_tensor_train, target_tensor_train, ans_sent_tensor_dev, ans_token_tensor_dev, target_tensor_dev, inp_tokenizer, targ_tokenizer = qg_dataset.load_dataset(
    max_length_ans_sent=max_length_inp, max_length_ans_token=10, max_vocab_inp=max_vocab_inp, max_length_targ=max_length_targ, max_vocab_targ=max_vocab_targ)

print("len ans_sent_tensor_train: ", len(ans_sent_tensor_train))
print("len ans_token_tensor_train: ", len(ans_token_tensor_train))
print("len target_tensor_train: ", len(target_tensor_train))

print("len input_tensor_dev: ", len(ans_sent_tensor_dev))
print("len input_tensor_dev: ", len(ans_token_tensor_dev))
print("len target_tensor_dev", len(target_tensor_dev))

print("Input Language; index to word mapping")
convert(inp_tokenizer, ans_sent_tensor_dev[0])
print()
print("Input Language; index to word mapping")
convert(inp_tokenizer, ans_token_tensor_dev[0])
print()
print("Target Language; index to word mapping")
convert(targ_tokenizer, target_tensor_dev[0])

embedding_dim = 300
if pretrained:
    inp_embedding_matrix = generate_embeddings_matrix(
        path_to_glove_file, inp_tokenizer, embedding_dim=embedding_dim)

    targ_embedding_matrix = generate_embeddings_matrix(
        path_to_glove_file, targ_tokenizer, embedding_dim=embedding_dim)        
else:
    inp_embedding_matrix = None
    targ_embedding_matrix = None

# Create a tf.data dataset
BUFFER_SIZE = len(ans_sent_tensor_dev)

steps_per_epoch = len(ans_sent_tensor_dev)//BATCH_SIZE
vocab_inp_size = len(inp_tokenizer.word_index)+1  # PADDING
vocab_tar_size = len(targ_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices(
    (ans_sent_tensor_train, ans_token_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

dataset_val = tf.data.Dataset.from_tensor_slices(
    (ans_sent_tensor_dev, ans_token_tensor_dev, target_tensor_dev))
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

example_ans_sent_batch, example_ans_token_batch, example_target_batch = next(iter(dataset))
tf.debugging.assert_shapes(
    [(example_ans_sent_batch, (BATCH_SIZE, max_length_inp))])
tf.debugging.assert_shapes(
    [(example_ans_token_batch, (BATCH_SIZE, 10))])
tf.debugging.assert_shapes(
    [(example_target_batch, (BATCH_SIZE, max_length_targ))])

example_ans_sent_batch_val, example_ans_token_batch_val,example_target_batch_val = next(iter(dataset_val))
tf.debugging.assert_shapes(
    [(example_ans_sent_batch_val, (BATCH_SIZE, max_length_inp))])
tf.debugging.assert_shapes(
    [(example_ans_token_batch_val, (BATCH_SIZE, 10))])
tf.debugging.assert_shapes(
    [(example_target_batch_val, (BATCH_SIZE, max_length_targ))])

# answer sentence encoder
ans_sent_encoder = Encoder(vocab_inp_size, embedding_dim, units,
                  bidirectional=bidirectional, embedding_matrix=inp_embedding_matrix,pretraine_embeddings=pretrained, layer=layer, dropout=dropout)

# target answer encoder
if answer_enc_units > 0:
    ans_token_encoder = Encoder(vocab_inp_size, embedding_dim, answer_enc_units,
                                bidirectional=bidirectional, embedding_matrix=inp_embedding_matrix, pretraine_embeddings=pretrained, layer=layer, dropout=dropout)
else:
    ans_token_encoder = None

# sample input
sample_output, sample_hidden = ans_sent_encoder(
    example_ans_sent_batch, training=True)
tf.debugging.assert_shapes(
    [(sample_output, (BATCH_SIZE, max_length_inp, units))])
if layer == 1:
    tf.debugging.assert_shapes([(sample_hidden, (BATCH_SIZE, units))])
else:
    tf.debugging.assert_shapes([(sample_hidden, (layer, BATCH_SIZE, units))])

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE,
                  targ_tokenizer.word_index['<start>'], targ_tokenizer.word_index['<end>'],  attention_type='luong', max_length_inp=max_length_inp, max_length_targ=max_length_targ, embedding_matrix=targ_embedding_matrix, pretraine_embeddings=pretrained, num_layers=layer, dropout=dropout)
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

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=path_to_logs, histogram_freq=1)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/model_{epoch}",
                                                         save_best_only=True,  # Only save a model if `val_loss` has improved.
                                                         monitor="val_loss",
                                                         verbose=1,
                                                         )

qg = QuestionGenerator(qg_dataset, inp_tokenizer, ans_sent_encoder,
                       decoder, targ_tokenizer, max_length_inp, ans_encoder=ans_token_encoder)
qg.compile(optimizer=optimizer, loss=loss_function)
# qg.build(tf.TensorShape((BATCH_SIZE, max_length_inp)))
# qg.summary()
qg.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[
       checkpoint_callback, 
       tensorboard_callback, early_stopping], validation_data=dataset_val)

# qg.save(path_to_model+"saved_model/")

qg.translate([['two', 'months', 'later', 'the', 'band', 'got', 'signed', 'to', 'a',
               'three', 'album', 'deal', 'with', 'spinefarm', ',', 'which', 'left', 'marko', 'displeased', '.']], [['spinefarm']], attention_plot_folder=path_to_model)
qg.translate([["Golm", "is", "a", "locality", "of", "Potsdam", ",", "the", "capital", "of",
             "the", "German", "state", "of", "Brandenburg", "."]], [["Potsdam"]], attention_plot_folder=path_to_model)
qg.translate(["the largest of these is the eldon square shopping centre , one of the largest city centre shopping complexes in the uk .".split(" ")], [["the", "eldon", "square", "shopping", "centre"]],
             attention_plot_folder=path_to_model)
qg.translate(["The name of the person is John .".split(" ")], [["John"]], attention_plot_folder=path_to_model)
qg.translate(["John is the name of the person .".split(" ")], [["John"]], attention_plot_folder=path_to_model)

qg.translate(
    ["the largest of these is the eldon square shopping centre , one of the largest city centre shopping complexes in the uk .".split(" ")], [["the", "eldon", "square", "shopping", "centre"]],beam_width=3)
qg.translate([['Golm', 'is', 'a', 'locality', 'of', 'Potsdam', ',', 'the',
              'capital', 'of', 'the', 'German', 'state', 'of', 'Brandenburg', '.']], [["Potsdam"]], beam_width=3)

dev_ans_sent, dev_ans_token, dev_questions = qg_dataset.create_dataset(qg_dataset.dev_path)
ans_sent_chunks = [dev_ans_sent[x:x+64] for x in range(0, len(dev_ans_sent), 64)]
ans_token_chunks = [dev_ans_token[x:x+64] for x in range(0, len(dev_ans_token), 64)]
for ans_sent_chunk, ans_token_chunk in zip(ans_sent_chunks, ans_token_chunks):
    outputs = qg.translate(ans_sent_chunk, ans_token_chunk, beam_width=3)
    filename = "dev.txt"

    with open(path_to_model + filename, "a") as f:
        for output in outputs:
            f.write(str(output))
            f.write('\n')

test_ans_sent, test_ans_token, test_questions = qg_dataset.create_dataset(qg_dataset.test_path)
ans_sent_chunks = [test_ans_sent[x:x+64] for x in range(0, len(test_ans_sent), 64)]
ans_token_chunks = [test_ans_token[x:x+64] for x in range(0, len(test_ans_token), 64)]
for ans_sent_chunk, ans_token_chunk in zip(ans_sent_chunks, ans_token_chunks):
    outputs = qg.translate(ans_sent_chunk, ans_token_chunk, beam_width=3)

    filename = "test.txt"

    with open(path_to_model + filename, "a") as f:
        for output in outputs:
            f.write(str(output))
            f.write('\n')

test_trf_ans_sent, test_trf_ans_token, test_trf_questions = qg_dataset.create_dataset(qg_dataset.test_trf_path)
ans_sent_chunks = [test_trf_ans_sent[x:x+64] for x in range(0, len(test_trf_ans_sent), 64)]
ans_token_chunks = [test_trf_ans_token[x:x+64] for x in range(0, len(test_trf_ans_token), 64)]
for ans_sent_chunk, ans_token_chunk in zip(ans_sent_chunks, ans_token_chunks):
    outputs = qg.translate(ans_sent_chunk, ans_token_chunk, beam_width=3)

    filename = "trf_test.txt"

    with open(path_to_model + filename, "a") as f:
        for output in outputs:
            f.write(str(output))
            f.write('\n')

if args.dataset == "squad":
    test_quac_ans_sent, test_quac_ans_token, test_quac_questions = qg_dataset.create_dataset(qg_dataset.test_quac_path)
    ans_sent_chunks = [test_quac_ans_sent[x:x+64] for x in range(0, len(test_quac_ans_sent), 64)]
    ans_token_chunks = [test_quac_ans_token[x:x+64] for x in range(0, len(test_quac_ans_token), 64)]
    for ans_sent_chunk, ans_token_chunk in zip(ans_sent_chunks, ans_token_chunks):
        outputs = qg.translate(ans_sent_chunk, ans_token_chunk, beam_width=3)

        filename = "quac_test.txt"

        with open(path_to_model + filename, "a") as f:
            for output in outputs:
                f.write(str(output))
                f.write('\n')
elif args.dataset == "quac":
    test_squad_ans_sent, test_squad_ans_token, test_squad_questions = qg_dataset.create_dataset(qg_dataset.test_squad_path)
    ans_sent_chunks = [test_squad_ans_sent[x:x+64] for x in range(0, len(test_squad_ans_sent), 64)]
    ans_token_chunks = [test_squad_ans_token[x:x+64] for x in range(0, len(test_squad_ans_token), 64)]
    for ans_sent_chunk, ans_token_chunk in zip(ans_sent_chunks, ans_token_chunks):
        outputs = qg.translate(ans_sent_chunk, ans_token_chunk, beam_width=3)

        filename = "squad_test.txt"

        with open(path_to_model + filename, "a") as f:
            for output in outputs:
                f.write(str(output))
                f.write('\n')
    


test_tedq_ans_sent, test_tedq_ans_token, test_tedq_questions = qg_dataset.create_dataset(qg_dataset.test_tedq_path)
ans_sent_chunks = [test_tedq_ans_sent[x:x+64] for x in range(0, len(test_tedq_ans_sent), 64)]
ans_token_chunks = [test_tedq_ans_token[x:x+64] for x in range(0, len(test_tedq_ans_token), 64)]
for ans_sent_chunk, ans_token_chunk in zip(ans_sent_chunks, ans_token_chunks):
    outputs = qg.translate(ans_sent_chunk, ans_token_chunk, beam_width=3)

    filename = "tedq_test.txt"

    with open(path_to_model + filename, "a") as f:
        for output in outputs:
            f.write(str(output))
            f.write('\n')