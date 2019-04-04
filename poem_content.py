import random

import numpy as np
from tensorflow import keras
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense

from peom_read import get_poem_name_pre2content

poem_pick = r"./poem/"
text_name, text_content, num2word, length = get_poem_name_pre2content(poem_pick + "poems.plk")
one_hot = np.eye(length + 1, dtype='int')
batch_size = 24


# def get_label(batch):
# train = np.array(one_hot[batch[0]])
# train = train.reshape([1, train.shape[0], train.shape[1]])
# print(train.shape)
# for i in range(batch.shape[0] - 1):
#     i += 1
#     tmp = one_hot[batch[i]]
#     # tmp = tmp.reshape([1, tmp.shape[0], tmp.shape[1]])
#     # # print(tmp.shape)
#     # train = np.r_[train, tmp]
# train = []
# for i in range(batch.shape[0]):
#     tmp = one_hot[batch[i]]
#     train.append(tmp)
# return np.array(train)
def get_label(batch):
    train = one_hot[batch]
    return train


# def get_input(batch):
#     # start = np.zeros(batch.shape[0], dtype='int')
#     # start[start == 0] = length + 1
#     start = []
#     for ii in range(batch.shape[0]):
#         start.append(np.insert(batch[ii], 0, [length + 1])[:-1])
#     start = np.array(start)
#     return start
def get_input(batch):
    batch = batch[:, :-1]
    start = np.zeros(batch.shape[0], dtype='int')
    start[start == 0] = length + 1
    return np.c_[start, batch]


class PoemSequence(keras.utils.Sequence):
    def __init__(self, encoder_data, decoder_data, batch_size_=32):
        self.encoder_data = encoder_data
        self.decoder_data = decoder_data
        self.batch_size = batch_size_

    def __len__(self):
        return len(self.encoder_data) // self.batch_size

    def __getitem__(self, idx):
        encoder_inp = self.encoder_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        decoder_inp = self.decoder_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        decoder_out = get_label(decoder_inp)
        decoder_inp = get_input(decoder_inp)
        return [encoder_inp, decoder_inp], decoder_out


# class PoemSequence_post(keras.utils.Sequence):
#     def __init__(self, encoder_data, decoder_data, batch_size_=32):
#         self.encoder_data = encoder_data[::-1]
#         self.decoder_data = decoder_data[::-1]
#         self.batch_size = batch_size_
#
#     def __len__(self):
#         return len(self.encoder_data) // self.batch_size
#
#     def __getitem__(self, idx):
#         encoder_inp = self.encoder_data[idx * self.batch_size:(idx + 1) * self.batch_size]
#         decoder_inp = self.decoder_data[idx * self.batch_size:(idx + 1) * self.batch_size]
#         decoder_out = get_label(decoder_inp)
#         decoder_inp = get_input(decoder_inp)
#         return [encoder_inp, decoder_inp], decoder_out


encoder_inputs = Input(shape=[None])
embedding_1 = Embedding(length + 1, 200)
x_ = embedding_1(encoder_inputs)
encoder = LSTM(200, return_state=True)
encoder_outputs, state_h, state_c = encoder(x_)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=[None])
x = Embedding(length + 2, 200)(decoder_inputs)
decoder = LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _, __ = decoder(x, initial_state=encoder_states)
decoder_dense = Dense(length + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
model.summary()
model.load_weights("./logs/poem_model_short/")

# embedding_weights = train_word2vec(text_name, num2word)
# weights = np.array([v for v in embedding_weights.values()])
# embedding_1.set_weights([weights])
# #
#
# model.compile(optimizer=keras.optimizers.SGD(lr=0.099, decay=0.0001, nesterov=True, momentum=0.99),
#               loss='categorical_crossentropy', )
# model.fit_generator(PoemSequence(text_name, text_content, batch_size), epochs=1, steps_per_epoch=200,
#                     callbacks=[
#                         ModelCheckpoint("./logs/poem_model_short/", save_best_only=True, period=1,
#                                         save_weights_only=True, monitor='loss'),
#                         TensorBoard("./logs/poem/", batch_size=batch_size, write_graph=True, write_grads=True),
#                         EarlyStopping(patience=4, baseline=0.1, monitor='loss')
#                     ])
#
# model.compile(optimizer=keras.optimizers.SGD(lr=0.01, decay=0.0001),
#               loss='categorical_crossentropy')
# model.fit_generator(PoemSequence(text_name, text_content, batch_size), epochs=1,steps_per_epoch=100,
#                     callbacks=[
#                         ModelCheckpoint("./logs/poem_model_short/", save_best_only=True, period=1,
#                                         save_weights_only=True, monitor='loss'),
#                         TensorBoard("./logs/poem/", batch_size=batch_size, write_graph=True, write_grads=True),
#                         EarlyStopping(patience=4, baseline=0.1, monitor='loss')
#                     ])
#
# model.compile(optimizer=keras.optimizers.SGD(lr=0.003, momentum=0.99, decay=1e-3,nesterov=True),
#               loss='categorical_crossentropy')
# model.fit_generator(PoemSequence(text_name, text_content, batch_size), epochs=1,
#                     callbacks=[
#                         ModelCheckpoint("./logs/poem_model_short/", save_best_only=True, period=1,
#                                         save_weights_only=True, monitor='loss'),
#                         TensorBoard("./logs/poem/", batch_size=batch_size, write_graph=True, write_grads=True),
#                         EarlyStopping(patience=4, baseline=0.1, monitor='loss')
#                     ])

encoder_model = keras.Model(inputs=[encoder_inputs], outputs=encoder_states)

decoder_states_h = Input([200])
decoder_states_c = Input([200])
decoder_states_input = [decoder_states_h, decoder_states_c]
decoder_outputs_, state_h_d, state_c_d = decoder(x, initial_state=decoder_states_input)
decoder_outputs_ = decoder_dense(decoder_outputs_)
decoder_model = keras.Model(
    [decoder_inputs, decoder_states_input[0], decoder_states_input[1]],
    [decoder_outputs_, state_h_d, state_c_d])


# decoder_model.summary()


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    # a = random.randint(0, length)
    target_seq = np.array([[length + 1]])
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict(
            [target_seq, states_value[0], states_value[1]])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0 or len(decoded_sentence) > 100:
            break
        sampled_char = num2word[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.

        # Update the target sequence (of length 1).
        target_seq = np.array([[sampled_token_index]])
        # Update states
        states_value = [h, c]
    return decoded_sentence


def name(text_na):
    sentence = ''
    for i in text_na:
        if i == 0:
            continue
        sentence += num2word[i]
    return sentence


for seq_index in range(5):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    a = random.randint(0, 5000)
    input_seq = text_name[a: a + 1]
    # input_seq = input_seq[input_seq != 0].reshape([1, -1])
    print('-')
    print('Input sentence:', name(text_name[a]))
    decoded_sentence = decode_sequence(input_seq)
    print('Decoded sentence:', decoded_sentence)
