import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


def generate_embeddings_matrix(path_to_glove_file, tokenizer, embedding_dim=300):
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found {} word vectors in {}.".format(
        len(embeddings_index), path_to_glove_file))

    hits = 0
    misses = 0

    # Prepare embedding matrix
    num_tokens = len(tokenizer.index_word) + 1
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for i, word in tokenizer.index_word.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and len(embedding_vector) == embedding_dim:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# function for plotting the attention weights


def plot_attention(attention, sentence, predicted_sentence):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    print("attention", attention.shape)
    print("sentence", sentence)
    print("predicted_sentence", predicted_sentence)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('books_read.png')
