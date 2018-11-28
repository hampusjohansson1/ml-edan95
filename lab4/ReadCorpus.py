from conll_dictorizer import CoNLLDictorizer, Token
from keras.preprocessing.text import Tokenizer
from keras import utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import scipy.spatial.distance as distance
import os
import bisect
import operator
from keras import Sequential
from keras import layers
from keras.models import load_model

BASE_DIR = '/usr/local/cs/EDAN95/datasets/'
glove_dir = './data/'

def load_conll2003_en():
    train_file = BASE_DIR + 'NER-data/eng.train'
    dev_file = BASE_DIR + 'NER-data/eng.valid'
    test_file = BASE_DIR + 'NER-data/eng.test'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

def load_data():
    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    test_dict = conll_dict.transform(test_sentences)
    dev_dict = conll_dict.transform(dev_sentences)

    return train_dict,test_dict,dev_dict

def load_glove():
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    return list(embeddings_index.keys()), embeddings_index

def extract_features(dict):
    X = []
    Y = []

    for sentence in dict:
        X_sentence = []
        Y_sentence = []

        for word in sentence:
            X_sentence.append(word.get("form").lower())
            Y_sentence.append(word.get("ner").lower())

        X.append(X_sentence)
        Y.append(Y_sentence)

    return X, Y

def extract_elements(sentences):
    extracted = set()

    for sentence in sentences:
        for e in sentence:
            extracted.add(e)

    return sorted(list(extracted))

def encode_to_indices(indices,input):
    encoded = []
    for arr in input:
        enc_elements = []
        for e in arr:
            if e in indices:
                enc_elements.append(indices[e])
            else:
                enc_elements.append(indices["word unknown"]) #Unknown word
        encoded.append(enc_elements)
    return encoded

def reverse_inidices(dictionary):
    inv_dict = {v: k for k, v in dictionary.items()}
    return inv_dict

def closest_neighbours(word, embeddings):
    word_embedding = embeddings[word]
    neighbours = {}
    for word, embedding in embeddings.items():
        similarity = distance.cosine(np.array(word_embedding), np.array(embedding))
        neighbours[word] = similarity
    sorted_list = sorted(neighbours.items(), key=operator.itemgetter(1), reverse=False)
    print(sorted_list[1:6])

def fill_glove_matrix(dictionary,embeddings):
    embedding_dim = 100
    max_words = len(dictionary)+2

    embedding_matrix = np.random.random((max_words, embedding_dim))
    for i, word in enumerate(dictionary):
        if word in embeddings:
            embedding_matrix[i] = embeddings[word]

    return embedding_matrix

#load all data necessary
train_dict,test_dict,dev_dict = load_data()
glove_words, embeddings = load_glove()

X_train, Y_train = extract_features(train_dict)
X_dev, Y_dev = extract_features(dev_dict)
X_test, Y_test = extract_features(test_dict)

#Extract all the words and categories in the test set and add to glove
X_words = extract_elements(X_train)
Y_cat = extract_elements(Y_train)
print("Our categories are:", Y_cat)

X_all_words = glove_words + X_words
dictionary = list(set(X_all_words)) #Remove duplicates
print("We have this many words in our dictionary: ", len(dictionary))

#Create the indices
X_indices = dict(enumerate(dictionary, start=2))
Y_indices = dict(enumerate(Y_cat, start=1))
X_indices[0], Y_indices[0] = '0', '0'
X_indices[1] = 'word unknown'
X_reverse = reverse_inidices(X_indices)
Y_reverse = reverse_inidices(Y_indices)

#Encode the train, dev and test set
X_train_encoded = encode_to_indices(X_reverse, X_train)
Y_train_encoded = encode_to_indices(Y_reverse, Y_train)
X_dev_encoded = encode_to_indices(X_reverse, X_dev)
Y_dev_encoded = encode_to_indices(Y_reverse, Y_dev)
X_test_encoded = encode_to_indices(X_reverse, X_test)
Y_test_encoded = encode_to_indices(Y_reverse, Y_test)

train_seq_length = max(len(s) for s in X_train_encoded)
dev_seq_length = max(len(s) for s in X_dev_encoded)
test_seq_length = max(len(s) for s in X_test_encoded)
max_seq_length = max(train_seq_length, dev_seq_length, test_seq_length)
print("Max Lenth:", max_seq_length)

#Fill the embedding matrix
embedding_matrix = fill_glove_matrix(dictionary, embeddings)
print("The matrix shape is: ", embedding_matrix.shape)

#Pad all the sequences
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_seq_length)
Y_train_padded = pad_sequences(Y_train_encoded, maxlen=max_seq_length)
Y_train_padded = utils.to_categorical(Y_train_padded)

X_dev_padded = pad_sequences(X_dev_encoded, maxlen=max_seq_length)
Y_dev_padded = pad_sequences(Y_dev_encoded, maxlen=max_seq_length)
Y_dev_padded = utils.to_categorical(Y_dev_padded)

X_test_padded = pad_sequences(X_test_encoded, maxlen=max_seq_length)
Y_test_padded = pad_sequences(Y_test_encoded, maxlen=max_seq_length)
Y_test_padded = utils.to_categorical(Y_test_padded)

#BUILD AND RUN MODELS
def build_RSS():
    model = Sequential()
    model.add(layers.Embedding(len(dictionary) + 2,
                               100,
                               mask_zero=True,
                               input_length=max_seq_length))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = True
    model.add(layers.SimpleRNN(32, return_sequences=True))
    model.add(layers.Dense(len(Y_cat) + 1, activation='softmax'))

    return model

def build_LSTM():
    model = Sequential()
    model.add(layers.Embedding(len(dictionary) + 2,
                               100,
                               mask_zero=True,
                               input_length=max_seq_length))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = True
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.Dense(len(Y_cat) + 1, activation='softmax'))
    return model


def train_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(X_train_padded, Y_train_padded, epochs=3, batch_size=64, verbose=1, validation_data=(X_dev_padded, Y_dev_padded))
    model.save("./model.h5")
    return model

#Run model
model = build_LSTM()
model = train_model(model)
#model = load_model("./model.h5")

loss, acc = model.evaluate(X_test_padded, Y_test_padded, batch_size=124, verbose=1)
print("acc: ", acc)
print("loss: ", loss)

Y_predicted = model.predict(X_test_padded)

#Tog din kod härifrån Adam pga. lite tidsbrist

#Remove padding
y_pred_probs_no_padd = []
for sent_nbr, sent_len_predictions in enumerate(Y_predicted):
    y_pred_probs_no_padd += [sent_len_predictions[-len(X_test[sent_nbr]):]]
print(y_pred_probs_no_padd[0])

#Get max probability and decode
y_pred = []
for sentence in y_pred_probs_no_padd:
    len_idx = list(map(np.argmax, sentence))
    len_cat = list(map(Y_indices.get, len_idx))
    y_pred += [len_cat]

