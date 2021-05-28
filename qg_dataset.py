import unicodedata
import pandas as pd

import ast

import tensorflow as tf

class QGDataset:
  def __init__(self, problem_type='squad', data_folder="/content/gdrive/MyDrive/mt-qg-data/01_data/preprocessedData/"):
    self.problem_type = problem_type
    self.dev_path = data_folder + problem_type + "/question_answer/dev.csv"
    self.train_path = data_folder + problem_type + "/question_answer/train.csv"
    self.test_path = data_folder + problem_type + "/question_answer/test.csv"
    self.inp_lang_tokenizer = None
    self.targ_lang_tokenizer = None

  # Converts the unicode file to ascii
  def unicode_to_ascii(self, s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
  
  def preprocess_sentence(self, w, include_eos_bos = True):
    w_result = []
    if include_eos_bos:
      w_result.append('<start>')
    for t in w:
      t = self.unicode_to_ascii(t.strip()) #.lower()

      # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
      #t = re.sub(r"[^a-zA-Z\d?.!,¿]+", " ", t)

      #t = t.strip()
      if t != '':
        w_result.append(t)
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    if include_eos_bos:
      w_result.append('<end>')
    return w_result

  # 1. Remove the accents
  # 2. Clean the sentences
  # 3. Return word pairs in the format: [answer_sentence, answer_token, question]
  def create_dataset(self, path):
    df = pd.read_csv(path)
    df["answer_sentence_token"] = [ast.literal_eval(t) for t in df["answer_sentence_token"]]
    df["question_token"] = [ast.literal_eval(t) for t in df["question_token"]]
    df["answer"] = [str(t).split(" ") for t in df["answer"]]

    #from plotly import express
    #display(express.histogram(x=[len(t) for t in df["answer_sentence_token"]]))

    #display(df["answer_sentence_token"].head(50))

    sentence_pairs = zip(df["answer_sentence_token"].apply(self.preprocess_sentence, include_eos_bos=False),
                         df["answer"].apply(self.preprocess_sentence, include_eos_bos=False),
                         df["question_token"].apply(self.preprocess_sentence))
    
    return zip(*sentence_pairs)

  def tokenize(self, lang, maxlen, max_vocab=None):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        #filters='',
        num_words=max_vocab,
        oov_token="<oov>")
    lang_tokenizer.word_index['<pad>'] = 0
    lang_tokenizer.index_word[0] = '<pad>'

    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                          maxlen=maxlen, padding="post")

    return tensor, lang_tokenizer

  def load_dataset(self, max_length_ans_sent, max_length_ans_token, max_vocab_inp, max_length_targ, max_vocab_targ):
    # creating cleaned input, output pairs
    ans_sent_train, ans_token_train, targ_lang_train = self.create_dataset(self.train_path)
    ans_sent_dev, ans_token_dev, targ_lang_dev = self.create_dataset(self.dev_path)

    print("ans_sent_train[-1]: ",ans_sent_train[-1])
    print("ans_token_train[-1]: ", ans_token_train[-1])
    print("targ_lang_train[-1]: ", targ_lang_train[-1])

    # TRAIN Tensors and Tokenizer
    ans_sent_tensor_train, inp_lang_tokenizer = self.tokenize(ans_sent_train, max_length_ans_sent, max_vocab=max_vocab_inp)
    target_tensor_train, targ_lang_tokenizer = self.tokenize(targ_lang_train, max_length_targ, max_vocab=max_vocab_targ)
    
    ans_token_tensor_train = inp_lang_tokenizer.texts_to_sequences(ans_token_train)
    ans_token_tensor_pad_train = tf.keras.preprocessing.sequence.pad_sequences(ans_token_tensor_train,maxlen=max_length_ans_token,padding='post')

    # DEV Tensors
    ans_sent_tensor_dev = inp_lang_tokenizer.texts_to_sequences(ans_sent_dev)
    ans_token_tensor_dev = inp_lang_tokenizer.texts_to_sequences(ans_token_dev)

    target_tensor_dev = targ_lang_tokenizer.texts_to_sequences(targ_lang_dev)

    print("ans_sent_tensor_dev[0]: ", ans_sent_tensor_dev[0])
    print("ans_token_tensor_dev[0]: ", ans_token_tensor_dev[0])
    print("target_tensor_dev[0] ", target_tensor_dev[0])

    # Padding Dev Tensor
    ans_sent_tensor_pad_dev = tf.keras.preprocessing.sequence.pad_sequences(ans_sent_tensor_dev,maxlen=max_length_ans_sent,padding='post')
    ans_token_tensor_pad_dev = tf.keras.preprocessing.sequence.pad_sequences(ans_token_tensor_dev,maxlen=max_length_ans_token,padding='post')
    target_tensor_pad_dev = tf.keras.preprocessing.sequence.pad_sequences(target_tensor_dev,maxlen=max_length_targ,padding='post')
    print("ans_sent_tensor_pad_dev[0]: ", ans_sent_tensor_pad_dev[0])
    print("ans_token_tensor_pad_dev[0]: ", ans_token_tensor_pad_dev[0])
    print("target_tensor_pad_dev[0] ", target_tensor_pad_dev[0])

    return ans_sent_tensor_train, ans_token_tensor_pad_train, target_tensor_train, ans_sent_tensor_pad_dev, ans_token_tensor_dev, target_tensor_pad_dev, inp_lang_tokenizer, targ_lang_tokenizer