from qg_dataset import QGDataset
from utils import convert, generate_embeddings_matrix
from encoder import Encoder
from decoder import Decoder

import tensorflow as tf

#PARAMS
path_to_folder = "/content/gdrive/MyDrive/mt-qg-data/01_data/preprocessedData/squad/question_answer/"
path_to_model = "/content/gdrive/MyDrive/mt-qg-data/00_models/qg_attention/squad/"
path_to_glove_file = "/content/gdrive/MyDrive/mt-qg-data/glove.840B.300d.txt"

max_length_targ = 20
max_length_inp = 80
max_vocab_inp = 45000
max_vocab_targ = 28000


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
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
units = 600
vocab_inp_size = len(inp_tokenizer.word_index)+1 # PADDING
vocab_tar_size = len(targ_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print("shape input_batch:", example_input_batch.shape)
print("shape target_batch:", example_target_batch.shape)

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