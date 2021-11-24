import pandas as pd
import numpy as np
import os

'''clear output'''
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True


def make_input_fn(X, y,NUM_EXAMPLES, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset

    return input_fn


class rf_class:
    def __init__(self,data,target,features_category,features_numeric,params):
        #self.data = data
        n = len(data)
        df = pd.DataFrame(np.random.randn(n, 2))
        msk = np.random.rand(n) < 0.8
        self.train = data[msk].copy()
        self.y_train = self.train.pop(target)
        self.test = data[~msk].copy()
        self.y_test = self.test.pop(target)
        self.NUM_EXAMPLES = len(self.train)

        def one_hot_cat_column(feature_name, vocab):
            return tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

        tf.random.set_seed(123)


        fc = tf.feature_column
        self.feature_columns = []
        for feature_name in features_category:
            # Need to one-hot encode categorical features.
            vocabulary = self.train[feature_name].unique()
            self.feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

        for feature_name in features_numeric:
            self.feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

        self.rf_classifier = tf.estimator.BoostedTreesClassifier(feature_columns=self.feature_columns, **params)

    def train(self,steps):
        print('Go')
        #train_input_fn = make_input_fn(self.train, self.y_train,NUM_EXAMPLES=self.NUM_EXAMPLES)
        #self.rf_classifier.train(train_input_fn, max_steps=steps)

    def eval(self):
        eval_input_fn = make_input_fn(self.test, self.y_test, shuffle=False, n_epochs=1,NUM_EXAMPLES=self.NUM_EXAMPLES)
        self.result = self.rf_classifier.evaluate(eval_input_fn)

    def print_eval(self):
        print(pd.Series(self.result))






