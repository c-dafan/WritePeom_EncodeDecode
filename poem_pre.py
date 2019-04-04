from os.path import exists

import numpy as np
from gensim.models import word2vec

np.random.seed(0)


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=200, min_word_count=1, context=5, name="embedding"):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    vocabulary_inv[0] = "\n"
    if "content" in name:
        vocabulary_inv[len(vocabulary_inv) + 1] = 'start'
    model_name = "./logs/models/" + name
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
    else:
        # Set values for various parameters
        num_workers = 4  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        # max_len = max(vocabulary_inv.keys()) + 1
        # vocabulary_inv[max_len] = 'end'

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()

        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights

# poem_pick = r"./data/poem/"
# text_name, text_content, num2word, length = get_poem_name2content(poem_pick + "poems.plk")
#
# num2word[length+1] = 'start'
# embedding_weights = train_word2vec(text_name, num2word, name="content_embedding")
#
# weights = np.array([v for v in embedding_weights.values()])
# print(weights.shape)
