import unicodedata
import pandas as pd

import ast

import tensorflow as tf

class QGDataset:
  def __init__(self, problem_type='squad', data_folder="/content/gdrive/MyDrive/mt-qg-data/01_data/preprocessedData/"):
    self.problem_type = problem_type
    self.dev_path = data_folder + problem_type + "/question_answer/dev.csv"
    self.train_path = data_folder + problem_type + "/question_answer/train.csv"
    self.inp_lang_tokenizer = None
    self.targ_lang_tokenizer = None

  # Converts the unicode file to ascii
  def unicode_to_ascii(self, s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
  
  def preprocess_sentence(self, w):
    w_result = ['<start>']
    for t in w:
      t = self.unicode_to_ascii(t.lower().strip())

      # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
      #t = re.sub(r"[^a-zA-Z\d?.!,¿]+", " ", t)

      #t = t.strip()
      if t != '':
        w_result.append(t)
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w_result.append('<end>')
    return w_result

  # 1. Remove the accents
  # 2. Clean the sentences
  # 3. Return word pairs in the format: [ENGLISH, SPANISH]
  def create_dataset(self, path):
    df = pd.read_csv(path)
    df["answer_sentence_token"] = [ast.literal_eval(t) for t in df["answer_sentence_token"]]
    df["question_token"] = [ast.literal_eval(t) for t in df["question_token"]]

    #from plotly import express
    #display(express.histogram(x=[len(t) for t in df["answer_sentence_token"]]))

    #display(df["answer_sentence_token"].head(50))

    sentence_pairs = zip(df["answer_sentence_token"].apply(self.preprocess_sentence), df["question_token"].apply(self.preprocess_sentence))
    

    return zip(*sentence_pairs)

  def tokenize(self, lang, maxlen, max_vocab):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        #filters='',
        num_words=max_vocab,
        oov_token="<oov>")
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                          maxlen=maxlen,
                                                          padding='post')

    return tensor, lang_tokenizer

  def load_dataset(self, max_length_inp, max_vocab_inp, max_length_targ, max_vocab_targ):
    # creating cleaned input, output pairs
    inp_lang_train, targ_lang_train = self.create_dataset(self.train_path)
    inp_lang_dev, targ_lang_dev = self.create_dataset(self.dev_path)

    print(inp_lang_train[-1])
    print(targ_lang_train[-1])

    input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang_train, max_length_inp, max_vocab_inp)
    target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang_train, max_length_targ, max_vocab_targ)

    print(inp_lang_dev[0])
    print(targ_lang_dev[0])
    input_tensor_dev = inp_lang_tokenizer.texts_to_sequences(inp_lang_dev)
    target_tensor_dev = targ_lang_tokenizer.texts_to_sequences(targ_lang_dev)

    input_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(input_tensor_dev,maxlen=max_length_inp,padding='post')
    target_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(target_tensor_dev,maxlen=max_length_targ,padding='post')

    print(input_tensor_dev[0])
    print(target_tensor_dev[0])

    return input_tensor, target_tensor, input_tensor_dev, target_tensor_dev, inp_lang_tokenizer, targ_lang_tokenizer