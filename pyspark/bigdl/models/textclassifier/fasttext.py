#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import itertools
import re
from optparse import OptionParser

from bigdl.dataset import news20
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample


def text_to_words(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words


def analyze_texts(data_rdd):
    def index(w_c_i):
        ((w, c), i) = w_c_i
        return (w, (i + 1, c))
    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0])) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda w_c: - w_c[1]).zipWithIndex() \
        .map(lambda w_c_i: index(w_c_i)).collect()


# pad([1, 2, 3, 4, 5], 0, 6)
def pad(l, fill_value, width):
    if len(l) >= width:
        return l[0: width]
    else:
        l.extend([fill_value] * (width - len(l)))
        return l


def build_model(seq_len, max_features, embedding_dim, class_num):
    model = Sequential()
    model.add(LookupTable(max_features, embedding_dim))
    model.add(Reshape([embedding_dim, 1, sequence_len]))
    model.add(SpatialConvolution(embedding_dim, 128, 5, 1))
    model.add(ReLU())
    model.add(SpatialMaxPooling(5, 1, 5, 1))
    model.add(SpatialConvolution(128, 128, 5, 1))
    model.add(ReLU())
    model.add(SpatialMaxPooling(5, 1, 5, 1))
    model.add(Reshape([128]))

    model.add(Linear(128, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model


def train(sc,
          batch_size,
          sequence_len, max_words, embedding_dim, training_split):
    print('Processing text dataset')
    texts = news20.get_news20()
    data_rdd = sc.parallelize(texts, 2)

    word_to_ic = analyze_texts(data_rdd)

    # # Only take the top wc between [10, sequence_len]
    # word_to_ic = dict(word_to_ic[10: max_words])

    # indexes = [v[0] for v in word_to_ic.values()]

    def create_ngram_set(input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.

        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for i in range(len(new_list) - ngram_range + 1):
                for ngram_value in range(2, ngram_range + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    X_ALL = [(item[0], item[1]) for item in data_rdd.collect()]
    X_all, y_all = zip(*X_ALL)
    all = []
    max_seqlength = 2000
    word_to_ic = dict(word_to_ic)
    for item in X_all:
        indexes_per_text = [word_to_ic[word][0] for word in text_to_words(item)]
        all.append(indexes_per_text)
    X_all = all
    ngram_range = 2
    max_features = np.hstack(all).max()
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in X_all:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}  # token to index map
    indice_token = {token_indice[k]: k for k in token_indice}  # only containing index

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting X_train and X_test with n-grams features
    X_all = add_ngram(X_all, token_indice, ngram_range)

    X_all = [pad(item, 0, max_seqlength) for item in X_all]

    def convertToOneHot(vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        if num_classes is None:
            num_classes = np.max(vector) + 1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)


    import random
    ss = zip(X_all, y_all)
    random.shuffle(ss)
    ssss = zip(*ss)
    tlen = int(len(ssss[0])*0.8)
    x_train = np.array(ssss[0][:tlen])
    y_train= convertToOneHot(np.array(ssss[1][:tlen]) - 1, 20)

    x_val = np.array(ssss[0][tlen:])
    y_val = convertToOneHot(np.array(ssss[1][tlen:]) - 1, 20)



    print('Build model...')
    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Embedding
    from keras.layers import GlobalAveragePooling1D
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Dense, Input, Flatten
    from keras.layers import Conv1D, MaxPooling1D, Embedding
    from keras.models import Model

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dim,
                        input_length=max_seqlength))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dense(20, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=128,
              nb_epoch=30,
              validation_data=(x_val, y_val))


    # sample_rdd = sc.parallelize(zip(X_all, y_all)).map(lambda item: Sample.from_ndarray(np.array(item[0]), np.array(item[1])))
    #
    # train_rdd, val_rdd = sample_rdd.randomSplit(
    #     [training_split, 1-training_split])
    # ttrain = train_rdd.collect()
    # optimizer = Optimizer(
    #     model=build_model(max_seqlength, max_features, embedding_dim, news20.CLASS_NUM),
    #     training_rdd=train_rdd,
    #     criterion=ClassNLLCriterion(),
    #     end_trigger=MaxEpoch(max_epoch),
    #     batch_size=batch_size,
    #     optim_method=Adagrad(learningrate=0.01, learningrate_decay=0.0002))
    #
    # optimizer.set_validation(
    #     batch_size=batch_size,
    #     val_rdd=val_rdd,
    #     trigger=EveryEpoch(),
    #     val_method=[Top1Accuracy()]
    # )
    # train_model = optimizer.optimize()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")  # noqa
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="15")
    parser.add_option("--model", dest="model_type", default="cnn")
    parser.add_option("-p", "--p", dest="p", default="0.0")

    (options, args) = parser.parse_args(sys.argv)
    if options.action == "train":
        batch_size = int(options.batchSize)
        embedding_dim = int(options.embedding_dim)
        max_epoch = int(options.max_epoch)
        p = float(options.p)
        model_type = options.model_type
        sequence_len = 50
        max_words = 1000
        training_split = 0.8
        sc = SparkContext(appName="text_classifier",
                          conf=create_spark_conf())
        init_engine()
        train(sc,
              batch_size,
              sequence_len, max_words, embedding_dim, training_split)
        sc.stop()
    elif options.action == "test":
        pass
