# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model

# std imports
import time
import gc
import os

from inputHandler import create_train_dev_set
from config import siamese_config


class Configuration(object):
    """Dump stuff here"""

CONFIG = Configuration()
CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']



def train_seimese_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='./'):
    """
    Train Siamese network to find similarity between sentences in `sentences1` and `sentences2`
        Steps Involved:
            1. Pass the each of sentences from sentences1 and sentences2 to bidirectional LSTM encoder.
            2. Merge the vectors from LSTM encodes and passed to dense layer.
            3. Pass the  dense layer vectors to sigmoid output layer.
            4. Use cross entropy loss to train weights
    Args:
        sentences_pair (list): list of tuple of sentence pairs
        is_similar (list): output value as respective sentences in
                            sentences1 and sentences2 are similar or not (1 if same else 0)
        embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
        model_save_directory (str): working directory for where to save models

    Returns:
        return (best_model_path):  path of best model
    """
    tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

    train_data_x1, train_data_x2, train_labels, leaks_train, \
    val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                           is_similar, CONFIG.max_sequence_length,
                                                                           CONFIG.validation_split_ratio)

    if train_data_x1 is None:
        print("++++ !! Failure: Unable to train model ++++")
        return None

    nb_words = len(tokenizer.word_index) + 1

    # Creating word embedding layer
    embedding_layer = Embedding(nb_words, CONFIG.embedding_dim, weights=[embedding_matrix],
                                input_length=CONFIG.max_sequence_length, trainable=False)

    # Creating LSTM Encoder
    lstm_layer = Bidirectional(LSTM(CONFIG.number_lstm_units, dropout=CONFIG.rate_drop_lstm, recurrent_dropout=CONFIG.rate_drop_lstm))

    # Creating LSTM Encoder layer for First Sentence
    sequence_1_input = Input(shape=(CONFIG.max_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    # Creating LSTM Encoder layer for Second Sentence
    sequence_2_input = Input(shape=(CONFIG.max_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    x2 = lstm_layer(embedded_sequences_2)

    # Creating leaks input
    leaks_input = Input(shape=(leaks_train.shape[1],))
    leaks_dense = Dense(CONFIG.number_dense_units/2, activation=CONFIG.activation_function)(leaks_input)

    # Merging two LSTM encodes vectors from sentences to
    # pass it to dense layer applying dropout and batch normalisation
    merged = concatenate([x1, x2, leaks_dense])
    merged = BatchNormalization()(merged)
    merged = Dropout(CONFIG.rate_drop_dense)(merged)
    merged = Dense(CONFIG.number_dense_units, activation=CONFIG.activation_function)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(CONFIG.rate_drop_dense)(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    STAMP = 'lstm_%d_%d_%.2f_%.2f' % (CONFIG.number_lstm_units, CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense)

    checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + STAMP + '.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

    tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

    model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
              validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
              epochs=200, batch_size=64, shuffle=True,
              callbacks=[early_stopping, model_checkpoint, tensorboard])

    return bst_model_path


def update_siamese_model(saved_model_path, new_sentences_pair, is_similar, embedding_meta_data):
    """
    Update trained siamese model for new given sentences1 and sentences2
        Steps Involved:
            1. Pass the each of sentences from sentences1 and sentences2 to bidirectional LSTM encoder.
            2. Merge the vectors from LSTM encodes and passed to dense layer.
            3. Pass the  dense layer vectors to sigmoid output layer.
            4. Use cross entropy loss to train weights
    Args:
        model_path (str): model path of already trained siamese model
        new_sentences_pair (list): list of tuple of new sentences pairs
        is_similar (list): output value as respective sentences in
                            sentences1 and sentences2 are similar or not (1 if same else 0)
        embedding_meta_data (dict): dict containing tokenizer and word embedding matrix

    Returns:
        return (best_model_path):  path of best model
    """
    tokenizer = embedding_meta_data['tokenizer']
    train_data_x1, train_data_x2, train_labels, leaks_train, \
    val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                           is_similar, MAX_SEQUENCE_LENGTH,
                                                                           VALIDATION_SPLIT)
    model = load_model(saved_model_path)
    model_file_name = saved_model_path.split('/')[-1]
    new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

    new_model_path = new_model_checkpoint_path + model_file_name
    model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                       save_best_only=True, save_weights_only=False)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))
    
    model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
              validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
              epochs=50, batch_size=3, shuffle=True,
              callbacks=[early_stopping, model_checkpoint, tensorboard])

    return new_model_path
