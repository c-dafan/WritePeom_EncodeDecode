from .peom_read import get_poem
from tensorflow.python.keras.layers import Embedding, LSTM
import tensorflow.python.keras as keras
import numpy as np

poem_pick = r"poem/"


def get_poem_model(word_num, embed_dim=200, input_dim=200):
    inputs = keras.Input([input_dim], )

    x = Embedding(word_num, embed_dim, )(inputs)
    x = LSTM(200, return_sequences=True)(x)
    x = keras.layers.Dense(word_num, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model


text, num2word, length = get_poem(poem_pick + "poems.plk")
one_hot = np.eye(length + 1, dtype='int')


# def get_label(row):
#     train = one_hot[row]
#     train_y = train[1:]
#     return row.reshape([-1, row.shape[0]]), np.concatenate([train_y, one_hot[0].reshape(1, train_y.shape[1])])
# def generate_arrays_from_data(data, batch_size):
#     length_ = data.shape[0]
#     pre = 0
#     while 1:
#         for row in data:
#             train, y_ = get_label(row)
#             yield train, y_.astype('int').reshape([-1, y_.shape[0], y_.shape[1]])


def get_label(batch):
    train = one_hot[batch]
    train_y = train[:, 1:]
    return batch, np.concatenate([train_y, one_hot[np.zeros([train_y.shape[0], 1], dtype='int')]], axis=1)


def generate_arrays_from_data(data, batch_size):
    length_ = data.shape[0]
    while 1:
        pre = 0
        for ind in range(batch_size, length_, batch_size):
            tex = data[pre:ind]
            pre = ind
            train, y_ = get_label(tex)
            yield train, y_.astype('int')


# def generate_arrays_from_file(path):
#     while 1:
#         f = open(path)
#         for line in f:
#             # 从文件中的每一行生成输入数据和标签的 numpy 数组，
#             x1, x2, y = process_line(line)
#             yield ({'input_1': x1, 'input_2': x2}, {'output': y})
#         f.close()
#
# model.fit_generator(generate_arrays_from_file('/my_file.txt'),
#                     steps_per_epoch=10000, epochs=10)

poem_model = get_poem_model(length + 1)
poem_model.summary()
poem_model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
poem_model.fit_generator(generate_arrays_from_data(text, 16), steps_per_epoch=400, epochs=32)
