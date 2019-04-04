import pickle
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def get_poem(path):
    with open(path, mode='rb') as poem:
        data_ = pickle.load(poem)
    text = data_['text']
    data = pad_sequences(text[:, 1], maxlen=199, padding='post')
    return (np.c_[data, np.zeros(data.shape[0])]).astype('int'), data_['num2word'], data_['len']


def get_poem_name2content(path):
    with open(path, mode='rb') as poem:
        data_ = pickle.load(poem)
    text = data_['text']
    test_content = pad_sequences(text[:, 1], maxlen=198, padding='post')
    text_name = pad_sequences(text[:, 0], maxlen=20, padding='post')
    test_content = (np.c_[test_content, np.zeros(test_content.shape[0], dtype='int')]).astype('int')
    test_content = (np.c_[test_content, np.zeros(test_content.shape[0], dtype='int')]).astype('int')
    return text_name, test_content, data_['num2word'], data_['len']


def get_poem_name_pre2content(path):
    with open(path, mode='rb') as poem:
        data_ = pickle.load(poem)
    text = data_['text']
    test_content = pad_sequences(text[:, 1], maxlen=118, padding='post')
    text_name = pad_sequences(text[:, 0], maxlen=20, padding='pre')
    test_content = (np.c_[test_content, np.zeros(test_content.shape[0], dtype='int')]).astype('int')
    test_content = (np.c_[test_content, np.zeros(test_content.shape[0], dtype='int')]).astype('int')
    return text_name, test_content, data_['num2word'], data_['len']


def get_poem_not_fix(path):
    """
    失败告终
    :param path:
    :return:
    """
    with open(path, mode='rb') as poem:
        data_ = pickle.load(poem)
    text = data_['text']
    text_name = text[:, 0]
    text_content = text[:, 1]
    text_content = np.array([np.append(ii, [0, 0]) for ii in text_content])
    return text_name, text_content, data_['num2word'], data_['len']
